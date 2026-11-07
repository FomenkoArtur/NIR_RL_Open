"""
rl_control.py - RL-регулятор в сценариях S1-S4 и эксперименте: тот же двойник (RLEnv, номинальный объект), что и при
обучении, поэтому GUI, таблица 1 и карты Шухарта считаются по тому же коду, что и PPO. Расчёт - на процессоре.
"""

import numpy as np
import torch

import rl_core as R
import rl_trainer
from config import G, N_PARAMS, T_START, T_TOTAL
from config_rl import RL
from episode import Episode
from fastsim import BatchEpisode, episode_noise
from scenarios import get_disturbance_profile

_CPU = torch.device('cpu')


class Controller:
    """Обученный RL-регулятор: сеть (актор + критик) и вид входа. Именно его хранит GUI."""

    def __init__(self, ac, feat=None):
        self.ac = ac.cpu().eval()
        self.feat = feat or RL['FEAT']

    def policy(self):
        return R.RLPolicy(self.ac, self.feat)


def load_controller(path=None):
    feat, ac = rl_trainer.load(path)
    return Controller(ac, feat)


class _ZeroPolicy:
    """Без регулятора: u = 0."""
    need_pid = False

    def reset(self, env):
        pass

    def act(self, env):
        return torch.zeros((env.B, N_PARAMS), dtype=env.dt, device=env.dev)


def _episodes(profiles, noises, t0s):
    nominal = R._plant(np.random.RandomState(0), RL['NOMINAL'])     # номинальный объект (как в РПЗ)
    cref = R._cref_eval()
    items = [dict(nominal, rate=rate, jump=jump, t0=int(t0), quiet=False, noise=noise, cref=cref, name='')
             for (rate, jump), noise, t0 in zip(profiles, noises, t0s)]
    return R.to_device(R._pack(items), _CPU)


@torch.no_grad()
def run_batch(profiles, noises, t0s, controller=None):
    """profiles - [(rate, jump)], noises - (B, T_TOTAL - 1, 8), t0s - такты скачка. -> BatchEpisode (X, N, K, eps, u)."""
    policy = controller.policy() if controller is not None else _ZeroPolicy()
    env = R.RLEnv(_CPU, reward=False, pid_inner=policy.need_pid)
    env.reset(_episodes(profiles, noises, t0s))
    policy.reset(env)
    X = torch.empty((env.B, T_TOTAL, N_PARAMS), dtype=env.dt)
    U = torch.zeros((env.B, T_TOTAL, N_PARAMS), dtype=env.dt)
    for t in range(T_TOTAL):
        X[:, t] = env.state                                        # критерии x_i(t) - для карт Шухарта
        U[:, t] = policy.act(env)
        if t < R.N_STEPS:                                          # на последнем такте команда уже ни на что не влияет
            env.step(U[:, t])
    eps = env.E.numpy()
    return BatchEpisode(X.numpy(), env.N.numpy(), G - eps, eps, U.numpy())


def run_scenario(scenario, seed, scale=1.0, controller=None, t0=T_START):
    """Один эпизод сценария S1-S4 (шум по seed, как раньше) -> Episode."""
    profile = get_disturbance_profile(scenario, t0, scale)
    batch = run_batch([profile], episode_noise(seed)[None], [t0], controller)
    return Episode(batch.X[0], batch.N[0], batch.K[0], batch.eps[0], batch.u[0])