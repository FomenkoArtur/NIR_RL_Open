"""
check_rl.py - проверка двух требований к RL-регулятору: (1) нет статической ошибки, (2) нет ложных срабатываний.

  python check_rl.py            weights: controller_rl.pt (или tuned/controller_rl.pt)

Для S1 (тренд x1), S2 (удар x6), S3 (скачок x8) и S4 (без возмущений), 30 реализаций, печатает:
  сист.ош.   систематическая статическая ошибка канала: среднее по эпизодам и последним 20 тактам от (N_i - g), со знаком
             возмущения (шум размаха +-0.03 по эпизодам усредняется; без регулятора ~ +0.45; цель |.| < 0.01);
  |ez| хвост среднее |N_i - g| за последние 20 тактов (включает шумовой пол ~0.01-0.03);
  FA эп.     доля эпизодов, где на КАНАЛАХ БЕЗ ВОЗМУЩЕНИЯ хоть раз |u| > 0.1 (цель 0); max|u| - наибольшее |u| на них.
Объекты: номинальный (шум x1) и рандомизированные (DR), для DR - отдельно по множителю шума (при x>3 шум сам даёт
отклонения до 0.4 в N, на таких объектах ложные срабатывания неизбежны для любого регулятора).
"""

import hw

hw.prepare()

import numpy as np
import torch

import rl_core as R
import rl_trainer
from config import G, IDLE_THRESHOLD, N_PARAMS, SIG_CUR, T_START, T_TOTAL, WINDOW
from config_rl import RL
from fastsim import episode_noise
from scenarios import get_disturbance_profile

SCEN = [('S1', 0), ('S2', 5), ('S3', 7), ('S4', None)]


def build(name, n, dr, seed0):
    rng = np.random.RandomState(seed0)
    rate, jump = get_disturbance_profile(name, T_START, 1.0)
    items = []
    for i in range(n):
        pl = R._plant(rng, RL['DR'] if dr else RL['NOMINAL'])
        noise = rng.normal(0.0, SIG_CUR, (T_TOTAL - 1, N_PARAMS)) * pl['ns'] if dr else episode_noise(seed0 + i)
        items.append(dict(pl, rate=rate, jump=jump, t0=T_START, quiet=(name == 'S4'), noise=noise,
                          cref=R._cref_eval(), act=R._act(rate, jump), name=name))
    return R._pack(items), rate, jump


def evaluate(ac, dev, dr, n=60, bins=None):
    policy = R.RLPolicy(ac.to(dev), RL['FEAT'])
    for name, ch in SCEN:
        ep, rate, jump = build(name, n, dr, 91000 + (500 if dr else 0))
        E, U, N = R.run_policy(policy, R.to_device(ep, dev), dev)
        N, U = N.cpu().numpy(), U.cpu().numpy()
        ns = np.asarray(ep['ns']).ravel()
        groups = [('все', np.ones(n, bool))] if not dr else [(f'шум x{lo}-{hi}', (ns >= lo) & (ns < hi)) for lo, hi in bins]
        for label, sel in groups:
            if not sel.any():
                continue
            if ch is None:
                fa = (np.abs(U[sel, WINDOW:]) > IDLE_THRESHOLD).any(axis=(1, 2)).mean()
                print(f"  {name} {label:14s} FA эп. {fa:5.2f}  max|u| {np.abs(U[sel, WINDOW:]).max():.3f}")
                continue
            sg = np.sign((rate.sum(0) + jump)[ch])
            ez = N[sel, -20:, ch] - G
            oth = [i for i in range(N_PARAMS) if i != ch]
            fa = (np.abs(U[sel][:, WINDOW:, oth]) > IDLE_THRESHOLD).any(axis=(1, 2)).mean()
            print(f"  {name} {label:14s} сист.ош. {sg * ez.mean():+.4f}   |ez| хвост {np.abs(ez).mean():.4f}   "
                  f"FA эп. {fa:5.2f}  max|u| {np.abs(U[sel][:, WINDOW:, oth]).max():.3f}")


def main():
    feat, ac = rl_trainer.load()
    dev = rl_trainer.get_device()
    print(f"Устройство: {dev}; head = {ac.head}, net = {ac.hp.get('net', 'conv')}")
    print("\nНОМИНАЛЬНЫЙ объект:")
    evaluate(ac, dev, False)
    print("\nРАНДОМИЗИРОВАННЫЕ объекты (a, c_i, задержка, шум, связи, мёртвая зона, насыщение):")
    evaluate(ac, dev, True, n=240, bins=[(0.5, 1.5), (1.5, 3.0), (3.0, 5.1)])


if __name__ == '__main__':
    main()