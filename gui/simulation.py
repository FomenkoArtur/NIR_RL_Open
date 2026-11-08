import numpy as np
from dataclasses import dataclass
import torch

from config import T_TOTAL, T_START, G
from plant import plant as twin_plant
from metrics import compute_metrics, out_of_control_rate
from scenarios import get_disturbance_profile
from gui.nomenclature import get_pid_gains, PLANT_PARAMS
from pid_control import PIDController
from mpc_control import MPCController
from rl_core import cref_features
from config_rl import RL


@dataclass
class EpisodeResult:
    X: np.ndarray
    N: np.ndarray
    K: np.ndarray
    eps: np.ndarray
    u: np.ndarray
    ts: int
    iae: float
    idle_rate: float
    oc_rate: float


def simulate_all(scenario, seed, scale, nom_key, rl_controller=None):
    np.random.seed(seed)

    p = PLANT_PARAMS[nom_key]
    a = p['a']
    # ТОЧНЫЙ ВЕКТОР эффективности привода (как в plant.py)
    c_vec = p['c'] * twin_plant.S

    pid = PIDController(**get_pid_gains(nom_key))
    mpc = MPCController(a=a, c=c_vec)  # Передаем вектор
    rl_policy = rl_controller.policy() if rl_controller else None

    rate, jump = get_disturbance_profile(scenario, T_START, scale)

    x = {"rl": twin_plant.x_bar.copy(), "pid": twin_plant.x_bar.copy(), "mpc": twin_plant.x_bar.copy()}
    norm = {"rl": twin_plant.make_normalizer(), "pid": twin_plant.make_normalizer(),
            "mpc": twin_plant.make_normalizer()}

    eps_win, n_win = [], []
    u_rl_hist = []
    cref = np.array(RL['REF']['EVAL_TAU'], dtype=float) if RL['REF']['MODE'] == 'tau' else np.array(
        RL['REF']['EVAL_PID'])
    cref_f = cref_features(torch.tensor(cref)).float().unsqueeze(0)

    res = {m: {'X': [], 'N': [], 'K': [], 'E': [], 'U': []} for m in ['rl', 'pid', 'mpc']}

    for t in range(T_TOTAL):
        if t == T_START:
            for m in x: x[m] += jump

        noise = np.random.normal(0, 0.0067, 8) if t > 0 else np.zeros(8)
        n_vals = {m: norm[m].update(x[m]) for m in x}

        # --- RL ---
        if rl_policy:
            eps_val = G - np.mean(n_vals['rl'])
            eps_win.append(eps_val)
            n_win.append(n_vals['rl'])
            if len(eps_win) > 32:
                eps_win.pop(0);
                n_win.pop(0)
            pad = 32 - len(eps_win)
            eps_p = [eps_win[0]] * pad + eps_win
            n_p = [n_win[0]] * pad + n_win

            n_arr = np.array(n_p).T
            features = torch.from_numpy((n_arr - 0.5) * 2.0).float().unsqueeze(0)
            with torch.no_grad():
                raw = rl_policy.ac.actor(features, cref_f)
                u_rl_raw = torch.tanh(raw).squeeze(0).numpy()

                # Сглаживание
                u_rl_hist.append(u_rl_raw)
                if len(u_rl_hist) > 3: u_rl_hist.pop(0)
                u_rl_smooth = np.mean(u_rl_hist, axis=0)

                # Мертвая зона 0.1
                dz = 0.1
                u_rl = np.where(np.abs(u_rl_smooth) < dz, 0.0, u_rl_smooth - np.sign(u_rl_smooth) * dz)
        else:
            u_rl = np.zeros(8)

        # --- PID и MPC ---
        e_pid = n_vals["pid"] - G
        u_pid = pid.act(e_pid, t)
        u_mpc = mpc.act(x["mpc"], t, rate[t])

        u = {"rl": u_rl, "pid": u_pid, "mpc": u_mpc}

        for m in x:
            res[m]['X'].append(x[m].copy())
            res[m]['N'].append(n_vals[m].copy())
            res[m]['K'].append(np.mean(n_vals[m]))
            res[m]['E'].append(G - np.mean(n_vals[m]))
            res[m]['U'].append(u[m].copy())

            if t < T_TOTAL - 1:
                # ИСПОЛЬЗУЕМ ВЕКТОР c_vec
                x[m] = (twin_plant.x_bar + a * (x[m] - twin_plant.x_bar) + rate[t]
                        - c_vec * u[m] + noise)

    out = {}
    for m in x:
        X = np.array(res[m]['X']);
        N = np.array(res[m]['N']);
        K = np.array(res[m]['K'])
        E = np.array(res[m]['E']);
        U = np.array(res[m]['U'])
        ts, iae, idle = compute_metrics(E, U, T_START)
        oc = out_of_control_rate(N, T_START)
        out[m] = EpisodeResult(X, N, K, E, U, ts, iae, idle, oc)

    return out