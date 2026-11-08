"""
rl_core.py - RL-контур: двойник с рандомизацией (пакет сред на устройстве), эталонная модель, награда,
сеть регулятора с настраиваемой архитектурой, ПИД на устройстве, PPO, метрики замкнутого контура и отчёт.
"""
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (A_CUR, C_RAW_FACTOR, CRITICAL_SCENARIOS, DELTA, G, IDLE_THRESHOLD, IQR_TO_SIGMA, N_CONS,
                    N_FEATURES, N_PARAMS, N_WINDOW, SCENARIOS, SEED_CRITICAL_START, SIG_CUR, T_OBS, T_START,
                    T_TOTAL, WINDOW)
from config_rl import RL, RL_ARCH
from fastsim import episode_noise
from plant import plant
from scenarios import critical_profile, custom_profile
from trainer import weighted_mse

X_BAR, S_SCALE = plant.x_bar, plant.S
LCL, UCL, S_MIN = plant.LCL, plant.UCL, plant.S_MIN
TREND, JUMP = SCENARIOS['S1']['rate_value'], abs(SCENARIOS['S2']['jump_value'])

D_MAX = 8
N_STEPS = T_TOTAL - 1
PRIV_DIM = 16
LOG2PI = math.log(2 * math.pi)

# ИСПРАВЛЕНИЕ: float32 для CUDA, чтобы не терять скорость
def _dt(dev):
    """Тип чисел объекта: double на CPU, float на CUDA/XPU (экономия скорости)."""
    return torch.float64 if dev.type == 'cpu' else torch.float32

def ref_dim():
    return 3 if RL['REF']['MODE'] == 'pid' else 2

def extra_dim(head):
    return ref_dim() + (N_PARAMS if head == 'pid_res' else 0)

def _val(rng, v, size=None):
    if isinstance(v, (tuple, list)):
        return rng.uniform(v[0], v[1], size)
    return np.full(size, float(v)) if size else float(v)

def _plant(rng, spec):
    d = spec['DELAY']
    d = int(rng.randint(d[0], d[1] + 1)) if isinstance(d, (tuple, list)) else int(d)
    cross = float(spec['CROSS'])
    off = np.zeros((N_PARAMS, N_PARAMS))
    if cross > 0:
        off = rng.uniform(-cross, cross, (N_PARAMS, N_PARAMS)) * (1.0 - np.eye(N_PARAMS))
    return {
        'a': _val(rng, spec['A']), 'c': _val(rng, spec['C_FRAC'], N_PARAMS) * S_SCALE,
        'd': d, 'ns': _val(rng, spec['NOISE']),
        'B': np.eye(N_PARAMS) + off, 'xm': float(np.abs(off).sum() / (N_PARAMS * (N_PARAMS - 1))),
        'dz': _val(rng, spec['DEAD']), 'slo': _val(rng, spec['SAT_LO']), 'shi': _val(rng, spec['SAT_HI']),
    }

def _disturbance(rng, a, dist):
    rate, jump = np.zeros((T_TOTAL, N_PARAMS)), np.zeros(N_PARAMS)
    kinds = list(dist['P_KIND'])
    p = np.array([dist['P_KIND'][k] for k in kinds], dtype=float)
    kind = kinds[rng.choice(len(kinds), p=p / p.sum())]
    t0 = int(rng.randint(dist['T0'][0], dist['T0'][1] + 1))
    par = int(rng.randint(N_PARAMS))
    sgn = float(rng.choice([-1.0, 1.0]))
    if kind == 'trend':
        rate[t0:, par] = sgn * rng.uniform(*dist['TREND'])
    elif kind == 'jump':
        jump[par] = sgn * rng.uniform(*dist['JUMP'])
    elif kind == 'pulse':
        dur = int(rng.randint(dist['PULSE_DUR'][0], dist['PULSE_DUR'][1] + 1))
        v = sgn * rng.uniform(*dist['PULSE'])
        rate[t0, par] += v
        if t0 + dur < T_TOTAL:
            rate[t0 + dur, par] -= v * a ** dur
    elif kind == 'sine':
        per = int(rng.randint(dist['SINE_PERIOD'][0], dist['SINE_PERIOD'][1] + 1))
        v = sgn * rng.uniform(*dist['SINE'])
        k = np.arange(T_TOTAL - t0)
        rate[t0:, par] += v * (2 * np.pi / per) * np.cos(2 * np.pi * k / per)
    elif kind == 'multi':
        n = int(rng.randint(dist['MULTI_N'][0], dist['MULTI_N'][1] + 1))
        idx = rng.choice(N_PARAMS, n, replace=False)
        jump[idx] = rng.choice([-1.0, 1.0], n) * rng.uniform(*dist['JUMP'], n) * dist['MULTI_SCALE']
    return rate * S_SCALE, jump * S_SCALE, t0, kind == 'none'

def _cref(rng):
    r = RL['REF']
    if r['MODE'] == 'pid':
        return rng.uniform(r['PID_SCALE'][0], r['PID_SCALE'][1], 3)
    return np.array([rng.uniform(*r['TAU']), rng.uniform(*r['OS'])])

def _cref_eval():
    r = RL['REF']
    return np.array(r['EVAL_PID'] if r['MODE'] == 'pid' else r['EVAL_TAU'], dtype=float)

def cref_features(cref):
    r = RL['REF']
    if r['MODE'] == 'pid':
        lo, hi = r['PID_SCALE']
        return (cref - (lo + hi) / 2) / ((hi - lo) / 2)
    lo = torch.tensor([r['TAU'][0], r['OS'][0]], dtype=cref.dtype, device=cref.device)
    hi = torch.tensor([r['TAU'][1], r['OS'][1]], dtype=cref.dtype, device=cref.device)
    return (cref - (lo + hi) / 2) / ((hi - lo) / 2)

def _pack(items):
    out = {k: np.stack([np.asarray(it[k]) for it in items]) for k in items[0] if k != 'name'}
    out['names'] = [it.get('name', '') for it in items]
    return out

def sample_episodes(rng, n, spec=None):
    spec = spec or RL['DR']
    items = []
    for _ in range(n):
        pl = _plant(rng, spec)
        rate, jump, t0, quiet = _disturbance(rng, pl['a'], RL['DIST'])
        noise = rng.normal(0.0, SIG_CUR, (T_TOTAL - 1, N_PARAMS)) * pl['ns']
        items.append(dict(pl, rate=rate, jump=jump, t0=t0, quiet=quiet, noise=noise, cref=_cref(rng), name='dr'))
    return _pack(items)

def suite_nominal():
    seed0 = RL['SEEDS']['NOMINAL']
    nom = _plant(np.random.RandomState(seed0), RL['NOMINAL'])
    items, k = [], 0
    for p in range(N_PARAMS):
        for kind, val in (('trend', TREND), ('jump', JUMP)):
            sign = 1 if (p + (kind == 'jump')) % 2 == 0 else -1
            rate, jump = custom_profile(p, kind, sign * val, T_START)
            items.append(dict(nom, rate=rate, jump=jump, t0=T_START, quiet=False, noise=episode_noise(seed0 + k),
                              cref=_cref_eval(), name=f'{kind} x{p + 1}'))
            k += 1
    for i in range(RL['SEARCH']['N_QUIET']):
        items.append(dict(nom, rate=np.zeros((T_TOTAL, N_PARAMS)), jump=np.zeros(N_PARAMS), t0=T_START, quiet=True,
                          noise=episode_noise(seed0 + 5000 + i), cref=_cref_eval(), name='S4'))
    return _pack(items)

def suite_dr(n=None, seed=None):
    n = n or RL['SEARCH']['N_EVAL_DR']
    ep = sample_episodes(np.random.RandomState(seed or RL['SEEDS']['EVAL']), n)
    ep['cref'][:] = _cref_eval()
    return ep

def suite_ood():
    rng = np.random.RandomState(RL['SEEDS']['OOD'])
    items = []
    for spec in RL['OOD']:
        full = dict(RL['NOMINAL'], **{k: v for k, v in spec.items() if k != 'name'})
        pl = _plant(rng, full)
        for _ in range(RL['SEARCH']['N_OOD_EP']):
            quiet = True
            while quiet:
                rate, jump, t0, quiet = _disturbance(rng, pl['a'], RL['DIST'])
            noise = rng.normal(0.0, SIG_CUR, (T_TOTAL - 1, N_PARAMS)) * pl['ns']
            items.append(dict(pl, rate=rate, jump=jump, t0=t0, quiet=False, noise=noise, cref=_cref_eval(),
                              name=spec['name']))
    return _pack(items)

def suite_critical():
    nom = _plant(np.random.RandomState(0), RL['NOMINAL'])
    items = []
    for ep, (name, spec) in enumerate(CRITICAL_SCENARIOS.items(), start=1):
        rate, jump = critical_profile(name)
        noise = episode_noise(SEED_CRITICAL_START + ep, spec.get('noise_scale', 1.0))
        items.append(dict(nom, rate=rate, jump=jump, t0=spec['t0'], quiet=False, noise=noise,
                          cref=_cref_eval(), name=name))
    return _pack(items)

def to_device(ep, dev):
    dt = _dt(dev)
    f = lambda k: torch.as_tensor(ep[k], dtype=dt, device=dev)
    out = {k: f(k) for k in ('c', 'B', 'rate', 'jump', 'noise', 'cref')}
    for k in ('a', 'ns', 'dz', 'slo', 'shi', 'xm'):
        out[k] = f(k).view(-1, 1)
    out['d'] = torch.as_tensor(ep['d'], dtype=torch.long, device=dev)
    out['t0'] = torch.as_tensor(ep['t0'], dtype=torch.long, device=dev)
    out['np'] = {'t0': np.asarray(ep['t0']), 'quiet': np.asarray(ep['quiet']), 'names': ep['names']}
    return out

class PIDTorch:
    def __init__(self, batch, dev, dt, kp, ki, kd):
        cfg = RL['PID']
        self.mode, self.imax, self.isep = cfg['mode'], cfg['I_MAX'], cfg['I_SEP']
        self.kp, self.ki, self.kd = kp, ki, kd
        self.n_i = int(cfg['WINDOW'])
        self.I = torch.zeros((batch, N_PARAMS), dtype=dt, device=dev)
        self.e_prev = torch.zeros((batch, N_PARAMS), dtype=dt, device=dev)
        self.hist = torch.zeros((batch, self.n_i, N_PARAMS), dtype=dt, device=dev)
        self.k = 0

    def act(self, x_norm):
        e = x_norm - G
        zero = torch.zeros_like(e)
        if self.mode == 'true':
            p = torch.where(e.abs() <= DELTA, zero, self.kp * e + self.kd * (e - self.e_prev))
            u_raw = p + self.I
            u = torch.clamp(u_raw, -1.0, 1.0)
            sat = ((u_raw > 1.0) & (e > 0)) | ((u_raw < -1.0) & (e < 0))
            freeze = sat | (e.abs() > self.isep)
            self.I = torch.clamp(self.I + torch.where(freeze, zero, self.ki * e), -self.imax, self.imax)
            self.e_prev = e
            return u
        pos = self.k % self.n_i
        self.hist[:, pos] = e
        i_term = torch.clamp(self.ki * self.hist.sum(dim=1), -self.imax, self.imax)
        u_raw = self.kp * e + self.kd * (e - self.e_prev) + i_term
        u = torch.clamp(u_raw, -1.0, 1.0)
        sat = ((u_raw > 1.0) & (e > 0)) | ((u_raw < -1.0) & (e < 0))
        self.hist[:, pos] = torch.where(sat, zero, e)
        self.e_prev, self.k = e, self.k + 1
        return torch.where(e.abs() <= DELTA, zero, u)

def pid_for(ep, dev, dt):
    cfg, batch = RL['PID'], ep['a'].shape[0]
    kp, ki, kd = cfg['KP'], cfg['KI'], cfg['KD']
    if RL['REF']['MODE'] == 'pid':
        sc = ep['cref']
        kp, ki, kd = kp * sc[:, 0:1], ki * sc[:, 1:2], kd * sc[:, 2:3]
    return PIDTorch(batch, dev, dt, kp, ki, kd)

class RLEnv:
    def __init__(self, dev, reward=True, pid_inner=False, shadow=False):
        self.dev, self.dt = dev, _dt(dev)
        self.reward_on, self.pid_inner, self.is_shadow = reward, pid_inner, shadow
        as_t = lambda v: torch.as_tensor(np.asarray(v), dtype=self.dt, device=dev)
        self.x_bar, self.lcl, self.ucl = as_t(X_BAR), as_t(LCL), as_t(UCL)
        self.s_min, self.S = as_t(S_MIN), as_t(S_SCALE)
        self.offsets = torch.arange(WINDOW, device=dev) - (WINDOW - 1)

    def reset(self, ep):
        dev, dt = self.dev, self.dt
        self.ep, self.B, self.t = ep, ep['a'].shape[0], 0
        b = self.B
        self.buf = torch.full((b, N_PARAMS, N_WINDOW), float('nan'), dtype=dt, device=dev)
        self.valid = torch.zeros((b, N_PARAMS), dtype=torch.long, device=dev)
        self.m = self.s = None
        self.state = self.x_bar.repeat(b, 1)
        self.H = torch.zeros((b, T_TOTAL + D_MAX, N_PARAMS), dtype=dt, device=dev)
        self.E = torch.empty((b, T_TOTAL), dtype=dt, device=dev)
        self.N = torch.empty((b, T_TOTAL, N_PARAMS), dtype=dt, device=dev)
        self.u_last = torch.zeros((b, N_PARAMS), dtype=dt, device=dev)
        self.held = torch.zeros(b, dtype=torch.long, device=dev)
        self.e_peak = torch.zeros(b, dtype=dt, device=dev)
        self.cref_f = cref_features(ep['cref']).float()
        if RL['REF']['MODE'] == 'tau':
            self.rho = torch.exp(-1.0 / ep['cref'][:, 0])
            self.os_ref = ep['cref'][:, 1]
        self.pid = pid_for(ep, dev, dt) if self.pid_inner else None
        self.u_pid = None
        self.shadow = None
        if self.reward_on and RL['REF']['MODE'] == 'pid' and not self.is_shadow:
            self.shadow = RLEnv(dev, reward=False, pid_inner=True, shadow=True)
            self.shadow.reset(ep)
        self._jump()
        self._observe()

    def _jump(self):
        hit = (self.ep['t0'] == self.t)[:, None]
        self.state = self.state + torch.where(hit, self.ep['jump'], torch.zeros_like(self.ep['jump']))

    def _observe(self):
        st, t = self.state, self.t
        admitted = ((st >= self.lcl) & (st <= self.ucl)) | (self.valid == 0)
        row = torch.where(admitted, st, torch.full_like(st, float('nan')))
        pos = t % N_WINDOW
        self.valid = self.valid - (~torch.isnan(self.buf[:, :, pos])).long()
        self.buf[:, :, pos] = row
        self.valid = self.valid + (~torch.isnan(row)).long()
        srt = torch.sort(self.buf, dim=2).values
        q = lambda r: torch.gather(srt, 2, torch.clamp(r - 1, min=0).unsqueeze(2)).squeeze(2)
        v = self.valid
        q25, q50, q75 = q((v + 3) // 4), q((v + 1) // 2), q((3 * v + 3) // 4)
        m_new, s_new = q50, torch.maximum((q75 - q25) / IQR_TO_SIGMA, self.s_min)
        if self.m is None:
            self.m, self.s = m_new, s_new
        else:
            empty = v == 0
            self.m, self.s = torch.where(empty, self.m, m_new), torch.where(empty, self.s, s_new)
        x_norm = 0.5 * (1.0 + torch.tanh(0.5 * (st - self.m) / self.s))
        self.N[:, t] = x_norm
        self.E[:, t] = G - x_norm.mean(dim=1)
        if self.pid is not None:
            self.u_pid = self.pid.act(x_norm)

    def window(self, feat=None):
        idx = torch.clamp(self.offsets + self.t, min=0)
        n = self.N[:, idx].float()
        return ((n - 0.5) * 2.0).transpose(1, 2)

    def extra(self, head):
        if head == 'pid_res':
            return torch.cat([self.cref_f, self.u_pid.float()], dim=1)
        return self.cref_f

    def priv(self):
        ep = self.ep
        p = torch.cat([(ep['a'] - 0.97) / 0.03,
                       (ep['c'] / self.S - 0.125) / 0.075,
                       (ep['d'].view(-1, 1).to(self.dt) - 2.0),
                       torch.log(ep['ns']),
                       ep['xm'] * 20.0,
                       ep['dz'] * 20.0,
                       ep['slo'], ep['shi'],
                       torch.full_like(ep['a'], self.t / T_TOTAL)], dim=1)
        return p.float()

    def step(self, u):
        u = u.to(self.dt)
        e_old = self.E[:, self.t]
        self.H[:, self.t + D_MAX] = u
        du = u - self.u_last
        self.u_last = u
        if self.shadow is not None:
            self.shadow.step(self.shadow.u_pid)
        self.t += 1
        t, ep = self.t, self.ep
        self._jump()
        idx = (t - ep['d'] + D_MAX).view(-1, 1, 1).expand(-1, 1, N_PARAMS)
        u_app = self.H.gather(1, idx).squeeze(1)
        u_act = torch.sign(u_app) * torch.relu(u_app.abs() - ep['dz'])
        u_act = torch.maximum(torch.minimum(u_act, ep['shi']), -ep['slo'])
        coupled = torch.bmm(ep['B'], u_act.unsqueeze(2)).squeeze(2)
        self.state = (self.x_bar + ep['a'] * (self.state - self.x_bar) + ep['rate'][:, t]
                      - ep['c'] * coupled + ep['noise'][:, t - 1])
        self._observe()
        if not self.reward_on:
            return None
        return self._reward(u, du, e_old)

    def _reward(self, u, du, e_old):
        rw, st, dt = RL['REWARD'], self.state, self.dt
        e = self.E[:, self.t]
        ae = e.abs()
        self.held = torch.where(ae <= DELTA, self.held + 1, torch.zeros_like(self.held))
        out = ((st > self.ucl) | (st < self.lcl)).to(dt).mean(1)
        mag = ((torch.relu(st - self.ucl) + torch.relu(self.lcl - st)) / self.S).mean(1)
        ez = self.N[:, self.t] - G
        zone = torch.relu(ez.abs() - DELTA).sum(1)
        r = (-rw['K_E'] * ae - rw['K_ZONE'] * zone - rw['K_OUT'] * out - rw['K_OUT_MAG'] * mag
             - rw['LAMBDA_U'] * (u ** 2).mean(1) - rw['LAMBDA_DU'] * (du ** 2).mean(1)
             + rw['BONUS'] * (self.held >= N_CONS).to(dt))
        if RL['REF']['MODE'] == 'tau':
            r = r - rw['LAMBDA_REF'] * torch.relu(ae - self.rho * e_old.abs() - DELTA)
            over = torch.relu(-torch.sign(self.e_peak) * e - self.os_ref * self.e_peak.abs() - DELTA)
            r = r - rw['LAMBDA_OS'] * over
            self.e_peak = torch.where(self.held >= N_CONS, torch.zeros_like(e),
                                      torch.where(ae > self.e_peak.abs(), e, self.e_peak))
        else:
            r = r - rw['LAMBDA_REF'] * torch.relu(ae - self.shadow.E[:, self.t].abs())
        return r.float()

def _norm(kind, ch):
    if kind == 'layer':
        return nn.GroupNorm(1, ch)
    if kind == 'group':
        return nn.GroupNorm(min(8, ch), ch)
    if kind == 'none':
        return nn.Identity()
    raise ValueError(f"norm={kind!r}: в RL допустимы 'layer', 'group', 'none'")

class RLNet(nn.Module):
    def __init__(self, hp, in_ch, extra_dim_, n_out):
        super().__init__()
        n = int(hp['n_conv'])
        chs = [in_ch] + [int(hp[f'c{i}']) for i in range(1, n + 1)]
        ks = [int(hp[f'k{i}']) for i in range(1, n + 1)]
        self.convs = nn.ModuleList([nn.Conv1d(chs[i], chs[i + 1], ks[i], padding=ks[i] // 2) for i in range(n)])
        self.norms = nn.ModuleList([_norm(hp['norm'], chs[i + 1]) for i in range(n)])
        c_last = chs[-1]
        self.film = None
        if hp.get('film') and extra_dim_ > 0:
            self.film = nn.Linear(extra_dim_, 2 * c_last)
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)
        self.pool = nn.AdaptiveAvgPool1d(int(hp['pool'])) if hp['temporal'] == 'avg' else None
        t_len = int(hp['pool']) if self.pool is not None else WINDOW
        self.skip = int(hp['skip'])
        self.z = int(hp['ctx_z'])
        self.ctx_type = hp.get('ctx_type', 'conv')
        if self.z and self.ctx_type == 'gru':
            self.ctx_rnn = nn.GRU(in_ch, 32, batch_first=True)
            self.ctx_out = nn.Linear(32, self.z)
        elif self.z:
            self.ctx_conv = nn.Sequential(nn.Conv1d(in_ch, 32, 5, padding=2), nn.ReLU(),
                                          nn.Conv1d(32, 32, 3, padding=1), nn.ReLU())
            self.ctx_out = nn.Linear(32, self.z)
        feat = c_last * t_len + in_ch * self.skip + self.z + extra_dim_
        self.fc1 = nn.Linear(feat, int(hp['fc']))
        self.fc2 = nn.Linear(int(hp['fc']), n_out)
        self.relu = nn.ReLU()

    def forward(self, x, extra):
        h = x
        for conv, norm in zip(self.convs, self.norms):
            h = self.relu(norm(conv(h)))
        if self.film is not None:
            g, b = self.film(extra).chunk(2, dim=1)
            h = h * (1.0 + g[:, :, None]) + b[:, :, None]
        if self.pool is not None:
            h = self.pool(h)
        parts = [h.flatten(1)]
        if self.skip:
            parts.append(x[:, :, -self.skip:].flatten(1))
        if self.z and self.ctx_type == 'gru':
            _, hn = self.ctx_rnn(x.transpose(1, 2))
            parts.append(torch.tanh(self.ctx_out(hn[-1])))
        elif self.z:
            parts.append(torch.tanh(self.ctx_out(self.ctx_conv(x).mean(dim=2))))
        if extra.shape[1]:
            parts.append(extra)
        return self.fc2(self.relu(self.fc1(torch.cat(parts, dim=1))))

class ActorCritic(nn.Module):
    def __init__(self, hp):
        super().__init__()
        self.hp, self.head = dict(hp), hp['head']
        e = extra_dim(self.head)
        self.priv_critic = RL['PPO']['PRIV_CRITIC']
        self.actor = RLNet(hp, N_FEATURES, e, N_PARAMS)
        self.critic = RLNet(hp, N_FEATURES, e + (PRIV_DIM if self.priv_critic else 0), 1)
        self.log_std = nn.Parameter(torch.full((N_PARAMS,), float(RL['PPO']['LOG_STD_INIT'])))
        nn.init.orthogonal_(self.actor.fc2.weight, 0.01)
        nn.init.zeros_(self.actor.fc2.bias)

    def value(self, x, extra, priv):
        z = torch.cat([extra, priv], dim=1) if self.priv_critic else extra
        return self.critic(x, z).squeeze(1)

    def n_actor_params(self):
        return sum(p.numel() for p in self.actor.parameters()) + self.log_std.numel()

def command(head, raw, u_prev, u_pid):
    h = RL['HEAD']
    if head == 'direct':
        return torch.tanh(raw)
    if head == 'delta':
        return torch.clamp(u_prev + h['DU_MAX'] * torch.tanh(raw), -1.0, 1.0)
    if head == 'pid_res':
        return torch.clamp(u_pid + h['RES_MAX'] * torch.tanh(raw), -1.0, 1.0)
    raise ValueError(f'head={head!r}')


def soft_deadzone(u, threshold=0.1):
    """Мягкая мёртвая зона: |u| <= threshold -> 0, иначе u сдвинута к нулю на threshold."""
    return torch.sign(u) * torch.relu(u.abs() - threshold)

def gauss_logp(raw, mu, log_std):
    return (-0.5 * ((raw - mu) / log_std.exp()) ** 2 - log_std - 0.5 * LOG2PI).sum(-1)

class RLPolicy:
    def __init__(self, ac, feat=None):
        self.ac, self.head, self.feat = ac.eval(), ac.head, feat or RL['FEAT']
        self.need_pid = self.head == 'pid_res'
    def reset(self, env):
        pass
    def act(self, env):
        raw = self.ac.actor(env.window(self.feat), env.extra(self.head))
        u = command(self.head, raw.to(env.dt), env.u_last, env.u_pid)
        return soft_deadzone(u, 0.1)

class PIDBaseline:
    need_pid = False
    def reset(self, env):
        c = RL['PID']
        self.pid = PIDTorch(env.B, env.dev, env.dt, c['KP'], c['KI'], c['KD'])
    def act(self, env):
        return self.pid.act(env.N[:, env.t])

class LegacyCNNPolicy:
    need_pid = False
    def __init__(self, model):
        self.model = model.eval()
    def reset(self, env):
        pass
    def act(self, env):
        return self.model(env.window()).to(env.dt)

@torch.no_grad()
def run_policy(policy, ep, dev):
    env = RLEnv(dev, reward=False, pid_inner=policy.need_pid)
    env.reset(ep)
    policy.reset(env)
    U = torch.zeros((env.B, T_TOTAL, N_PARAMS), dtype=env.dt, device=dev)
    for t in range(N_STEPS):
        U[:, t] = policy.act(env)
        env.step(U[:, t])
    U[:, -1] = policy.act(env)
    return env.E, U, env.N

def cl_metrics(E, U, N, ep_np):
    m = RL['METRIC']
    t0s, quiet, names = ep_np['t0'], ep_np['quiet'], ep_np['names']
    rows = []
    for b in range(len(t0s)):
        t0 = int(t0s[b])
        e, u = E[b, t0:t0 + T_OBS], U[b, t0:t0 + T_OBS]
        ez = N[b, t0:t0 + T_OBS] - G
        n = len(e)
        if quiet[b]:
            rows.append({'name': names[b], 'quiet': True, 'FA': float(np.mean(np.abs(u).max(axis=1) > IDLE_THRESHOLD))})
            continue
        ok = np.abs(e) <= DELTA
        settled = (np.lib.stride_tricks.sliding_window_view(ok, N_CONS).all(axis=1)
                   if n >= N_CONS else np.zeros(1, dtype=bool))
        sign = np.sign(e[1:6].mean())
        os_ = max(0.0, float((-sign * e[m['OS_FROM']:]).max())) if sign != 0 and n > m['OS_FROM'] else 0.0
        rows.append({'name': names[b], 'quiet': False,
                     'IAE': float(np.abs(e).sum()),
                     'TS': float(settled.argmax()) if settled.any() else float(n),
                     'OS': os_,
                     'EN': float((u ** 2).sum(axis=1).mean()),
                     'ZE': float(np.maximum(np.abs(ez) - DELTA, 0.0).sum(axis=1).mean())})
    return _aggregate(rows)

def _aggregate(rows, by_name=True):
    m = RL['METRIC']
    dist = [r for r in rows if not r['quiet']]
    qt = [r for r in rows if r['quiet']]
    mean = lambda rs, k: float(np.mean([r[k] for r in rs])) if rs else 0.0
    o = {k: mean(dist, k) for k in ('IAE', 'TS', 'OS', 'EN', 'ZE')}
    o['FA'] = mean(qt, 'FA')
    o['J'] = (m['W_IAE'] * o['IAE'] + m['W_TS'] * o['TS'] / T_OBS + m['W_OS'] * o['OS']
              + m['W_EN'] * o['EN'] + m['W_FA'] * o['FA'] + m['W_ZE'] * o['ZE'])
    if by_name:
        o['by_name'] = {g: _aggregate([r for r in rows if r['name'] == g], False)
                        for g in dict.fromkeys(r['name'] for r in rows)}
    return o

def rl_evaluate(policy, ep, dev):
    E, U, N = run_policy(policy, ep, dev)
    return cl_metrics(E.cpu().numpy(), U.cpu().numpy(), N.cpu().numpy(), ep['np'])

def make_suites(dev, which=('nominal', 'dr', 'ood', 'critical')):
    build = {'nominal': suite_nominal, 'dr': suite_dr, 'ood': suite_ood, 'critical': suite_critical}
    return {k: to_device(build[k](), dev) for k in which}

class PPO:
    def __init__(self, arch, train_hp, dev, seed, log=print):
        self.dev, self.cfg, self.log = dev, RL['PPO'], log
        torch.manual_seed(seed)
        self.rng = np.random.RandomState(seed)
        self.ac = ActorCritic(arch).to(dev)
        self.head, self.feat = arch['head'], RL['FEAT']
        self.n_env = int(self.cfg['N_ENVS'].get(dev.type, 64))
        self.env = RLEnv(dev, reward=True, pid_inner=self.head == 'pid_res')
        self.cuda = dev.type == 'cuda'
        lr0 = float(train_hp['LR'])
        self.params = list(self.ac.parameters())
        self.opt = optim.Adam(self.params, lr=torch.tensor(lr0, device=dev) if self.cuda else lr0,
                              weight_decay=float(train_hp['WEIGHT_DECAY']),
                              capturable=self.cuda, foreach=self.cuda or None)
        s, b, e = N_STEPS, self.n_env, extra_dim(self.head)
        z = lambda *shape: torch.zeros(shape, device=dev)
        self.obs, self.ext, self.prv = z(s, b, N_FEATURES, WINDOW), z(s, b, e), z(s, b, PRIV_DIM)
        self.raw, self.logp, self.val = z(s, b, N_PARAMS), z(s, b), z(s, b)
        self.rew, self.adv, self.ret = z(s, b), z(s, b), z(s, b)
        n = s * b
        self.f = {'obs': self.obs.view(n, N_FEATURES, WINDOW), 'ext': self.ext.view(n, e),
                  'prv': self.prv.view(n, PRIV_DIM), 'raw': self.raw.view(n, N_PARAMS),
                  'logp': self.logp.view(n), 'adv': self.adv.view(n), 'ret': self.ret.view(n)}
        self.mb = min(int(train_hp['BATCH_SIZE']), n)
        self.nb = n // self.mb
        self.idx2d = torch.randperm(n, device=dev)[:self.nb * self.mb].view(self.nb, self.mb).clone()
        self.k = torch.zeros(1, dtype=torch.long, device=dev)
        self.pg_coef = torch.ones((), device=dev)
        self.kl_acc = torch.zeros((), device=dev)
        gru = int(arch.get('ctx_z', 0)) > 0 and arch.get('ctx_type') == 'gru'
        self.use_graph = self.cuda and self.cfg['CUDA_GRAPH'] and not gru
        self.graph, self.iter, self.vf_warmup = None, 0, 0

    def pretrain_imitation(self):
        cfg, dev = RL['IMIT'], self.dev
        env = RLEnv(dev, reward=False, pid_inner=True)
        X, EX, UP, UPID, Y = [], [], [], [], []
        left = int(cfg['EPISODES'])
        with torch.no_grad():
            while left > 0:
                nb = min(self.n_env, left)
                ep = to_device(sample_episodes(self.rng, nb, RL['DR'] if cfg['DR'] else RL['NOMINAL']), dev)
                env.reset(ep)
                for t in range(N_STEPS):
                    X.append(env.window(self.feat))
                    EX.append(env.extra(self.head))
                    UP.append(env.u_last.float())
                    UPID.append(env.u_pid.float())
                    Y.append(env.u_pid.float())
                    env.step(env.u_pid)
                left -= nb
        X, EX, UP, UPID, Y = (torch.cat(a) for a in (X, EX, UP, UPID, Y))
        opt = optim.Adam(self.ac.actor.parameters(), lr=float(cfg['LR']))
        n, bs = len(X), int(cfg['BATCH'])
        for epoch in range(int(cfg['EPOCHS'])):
            perm = torch.randperm(n, device=dev)
            total = torch.zeros((), device=dev)
            for i in range(0, n - bs + 1, bs):
                idx = perm[i:i + bs]
                u = command(self.head, self.ac.actor(X[idx], EX[idx]), UP[idx], UPID[idx])
                loss = weighted_mse(u, Y[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 1.0)
                opt.step()
                total += loss.detach()
            if epoch == 0 or (epoch + 1) % 5 == 0:
                self.log(f"  имитация: эпоха {epoch + 1}, loss = {total.item() / max(1, n // bs):.5f}")
        with torch.no_grad():
            self.ac.log_std.fill_(float(cfg['LOG_STD_AFTER']))
        self.vf_warmup = int(cfg['VF_WARMUP_ITERS'])

    @torch.no_grad()
    def collect(self):
        env, ac, g, lam = self.env, self.ac, self.cfg['GAMMA'], self.cfg['LAMBDA']
        env.reset(to_device(sample_episodes(self.rng, self.n_env), self.dev))
        ac.eval()
        std = ac.log_std.exp()
        for t in range(N_STEPS):
            x, e, p = env.window(self.feat), env.extra(self.head), env.priv()
            mu = ac.actor(x, e)
            raw = mu + std * torch.randn_like(mu)
            self.obs[t], self.ext[t], self.prv[t], self.raw[t] = x, e, p, raw
            self.logp[t] = gauss_logp(raw, mu, ac.log_std)
            self.val[t] = ac.value(x, e, p)
            self.rew[t] = env.step(soft_deadzone(command(self.head, raw.to(env.dt), env.u_last, env.u_pid), 0.1))
        last = ac.value(env.window(self.feat), env.extra(self.head), env.priv())
        a = torch.zeros(self.n_env, device=self.dev)
        for t in reversed(range(N_STEPS)):
            nv = last if t == N_STEPS - 1 else self.val[t + 1]
            a = self.rew[t] + g * nv - self.val[t] + g * lam * a
            self.adv[t] = a
        self.ret.copy_(self.adv + self.val)
        return self.rew.sum(0).mean()

    def _step(self):
        f, c = self.f, self.cfg
        idx = self.idx2d.index_select(0, self.k).view(-1)
        x, e, p = f['obs'].index_select(0, idx), f['ext'].index_select(0, idx), f['prv'].index_select(0, idx)
        raw, lp_old = f['raw'].index_select(0, idx), f['logp'].index_select(0, idx)
        adv, ret = f['adv'].index_select(0, idx), f['ret'].index_select(0, idx)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        lp = gauss_logp(raw, self.ac.actor(x, e), self.ac.log_std)
        ratio = torch.exp(lp - lp_old)
        pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - c['CLIP'], 1 + c['CLIP']) * adv).mean()
        vf = ((self.ac.value(x, e, p) - ret) ** 2).mean()
        ent = (self.ac.log_std + 0.5 * (LOG2PI + 1.0)).sum()
        loss = self.pg_coef * pg + c['VF_COEF'] * vf - c['ENT_COEF'] * ent
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, c['MAX_GRAD'])
        self.opt.step()
        self.kl_acc.add_((lp_old - lp).mean().detach())
        self.k.add_(1)

    def _runner(self):
        if not self.use_graph:
            return self._step
        if self.graph is None:
            snap = {k: v.clone() for k, v in self.ac.state_dict().items()}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self._step()
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._step()
            with torch.no_grad():
                for k, v in self.ac.state_dict().items():
                    v.copy_(snap[k])
                for st in self.opt.state.values():
                    for v in st.values():
                        v.zero_()
        return self.graph.replay

    def update(self):
        run, n, kl = self._runner(), self.f['obs'].shape[0], 0.0
        for _ in range(int(self.cfg['EPOCHS'])):
            self.idx2d.copy_(torch.randperm(n, device=self.dev)[:self.nb * self.mb].view(self.nb, self.mb))
            self.k.zero_()
            self.kl_acc.zero_()
            self.ac.train()
            for _ in range(self.nb):
                run()
            kl = self.kl_acc.item() / self.nb
            if kl > self.cfg['TARGET_KL']:
                break
        return kl

    def iterate(self):
        ret = self.collect()
        self.pg_coef.fill_(0.0 if self.iter < self.vf_warmup else 1.0)
        kl = self.update()
        self.iter += 1
        return {'ret': ret.item(), 'kl': kl}

SUITE_TITLES = {'nominal': 'Номинальный объект', 'dr': 'Рандомизированные объекты (обучающие диапазоны)',
                'ood': 'Объекты вне обучающих диапазонов', 'critical': 'Критические ситуации C1-C9 (отложенная проверка)'}

def _ms(vals):
    return float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else 0.0

def print_report(results, log=print):
    keys = ('IAE', 'TS', 'OS', 'EN', 'ZE', 'FA', 'J')
    for suite, title in SUITE_TITLES.items():
        rows = [(n, r[suite]) for n, r in results.items() if suite in r]
        if not rows:
            continue
        log(f"\n--- {title} ---")
        log(f"{'регулятор':<28}" + ''.join(f"{k:>16}" for k in keys))
        for name, runs in rows:
            cells = []
            for k in keys:
                mu, sd = _ms([m[k] for m in runs])
                cells.append(f"{mu:>9.3f}" + (f" ±{sd:<5.3f}" if sd else ' ' * 7))
            log(f"{name:<28}" + ''.join(cells))
        if suite in ('critical', 'ood'):
            names = list(rows[0][1][0].get('by_name', {}))
            if names:
                log(f"{'J по ' + ('ситуациям' if suite == 'critical' else 'объектам'):<28}"
                    + ''.join(f"{s[:10]:>11}" for s in names))
                for name, runs in rows:
                    log(f"{name:<28}" + ''.join(f"{_ms([m['by_name'][s]['J'] for m in runs])[0]:>11.3f}" for s in names))
