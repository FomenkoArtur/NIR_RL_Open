#!/usr/bin/env python
"""
Визуальная диагностика RL-агента.
1. Обучает короткий прогон PPO (iters) на текущих настройках reward.
2. Симулирует S1, S2, S3 с новыми весами.
3. Сохраняет графики в diagnostics/ и печатает числовой вердикт:
   - статическая ошибка на S1 (должна быть < 0.01)
   - макс. ложное срабатывание на S1/S2/S3 (должно быть < 0.05)

Возвращает 0 если всё OK, 2 если есть проблемы, 1 если ошибка.
"""
import sys
from pathlib import Path
import numpy as np

# ВАЖНО: hw.prepare() до импорта torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hw
hw.prepare()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import T_START, G
from config_rl import RL, RL_ARCH, RL_TRAIN


def train_short(iters, seed=42):
    from rl_trainer import train_rl, save
    print(f"[diagnose] обучение {iters} итераций...", flush=True)
    ac = train_rl(arch=dict(RL_ARCH), train_hp=dict(RL_TRAIN), seed=seed,
                  iters=iters, log=lambda m: print(f"  {m}", flush=True), validate=False)
    save(ac, "controller_rl.pt")
    print("[diagnose] веса сохранены в controller_rl.pt", flush=True)


def diagnose(weights="controller_rl.pt"):
    from rl_control import load_controller
    from gui.simulation import simulate_all

    controller = load_controller(weights)
    out_dir = Path("diagnostics")
    out_dir.mkdir(exist_ok=True)

    disturbed = {'S1': 0, 'S2': 5, 'S3': 7}
    results = {}

    for sc in ['S1', 'S2', 'S3']:
        res = simulate_all(sc, seed=42, scale=1.0, nom_key="shaft", rl_controller=controller)
        rl = res['rl']

        # -------- График --------
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        axes[0].plot(rl.K, color='tab:blue', lw=2, label='RL K(t)')
        axes[0].axhline(G, color='k', ls='--', alpha=0.5)
        axes[0].axhline(G + 0.03, color='g', ls=':', alpha=0.5)
        axes[0].axhline(G - 0.03, color='g', ls=':', alpha=0.5)
        axes[0].axvline(T_START, color='r', ls=':')
        axes[0].set_ylabel('K(t)')
        axes[0].set_title(f'{sc}: RL-регулятор (seed 42)')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        for i in range(8):
            axes[1].plot(rl.u[:, i], label=f'u(x{i+1})', alpha=0.7)
        axes[1].axhline(0.1, color='orange', ls='--', label='порог Х.Х.')
        axes[1].axhline(-0.1, color='orange', ls='--')
        axes[1].set_ylabel('u(t)')
        axes[1].legend(ncol=4, fontsize='small')
        axes[1].grid(alpha=0.3)

        for i in range(8):
            axes[2].plot(rl.N[:, i], label=f'x{i+1}', alpha=0.7)
        axes[2].axhline(0.5, color='k', ls='--', alpha=0.5)
        axes[2].set_ylabel('N(t)')
        axes[2].set_xlabel('Такт')
        axes[2].legend(ncol=4, fontsize='small')
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(out_dir / f"{sc}_rl.png", dpi=100)
        plt.close()

        # -------- Метрики --------
        # Статическая ошибка: среднее |K - g| на последних 30 тактах
        static_err = float(np.abs(rl.K[-30:] - G).mean())

        # Ложные срабатывания: макс |u| на НЕвозмущённых каналах после t0
        u_after = rl.u[T_START:]
        fa_max = 0.0
        for ch in range(8):
            if ch != disturbed[sc]:
                fa_max = max(fa_max, float(np.abs(u_after[:, ch]).max()))

        results[sc] = {'static': static_err, 'fa': fa_max}

    # -------- Вердикт --------
    print("\n=== ОТЧЁТ ВИЗУАЛЬНОЙ ДИАГНОСТИКИ ===", flush=True)
    ok_all = True
    for sc, r in results.items():
        bad_static = r['static'] > 0.01
        bad_fa = r['fa'] > 0.05
        mark = "OK" if not (bad_static or bad_fa) else "ПЛОХО"
        print(f"{sc}: статика={r['static']:.4f} {'(OK)' if not bad_static else '(>0.01)'} | "
              f"ложные={r['fa']:.4f} {'(OK)' if not bad_fa else '(>0.05)'} [{mark}]", flush=True)
        if bad_static or bad_fa:
            ok_all = False

    if ok_all:
        print("ИТОГ: ВСЁ ХОРОШО (статика и ложные срабатывания в норме)", flush=True)
        return 0
    else:
        print("ИТОГ: ЕСТЬ ПРОБЛЕМЫ. Модели нужно исправить функцию награды.", flush=True)
        return 2


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=300, help="итераций обучения")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--weights", default="controller_rl.pt")
    args = ap.parse_args()

    try:
        if not args.skip_train:
            train_short(args.iters)
        rc = diagnose(args.weights)
        return rc
    except Exception as e:
        import traceback
        print(f"ОШИБКА: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
