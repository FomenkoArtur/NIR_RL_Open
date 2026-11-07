"""
Главный скрипт для проведения вычислительного эксперимента.
Апробация контура автоматического управления с RL-регулятором (Глава 3).

Запуск:  python main.py               обучить RL-регулятор (PPO), сохранить controller_rl.pt, провести эксперимент
         python main.py --load        взять готовые веса (controller_rl.pt или tuned/controller_rl.pt), без обучения
         python main.py --iters 100   другое число итераций PPO
"""

import argparse

import hw

hw.prepare()

from rl_trainer import load, save, train_rl
from config_rl import RL
from experiment import run_full_experiment
from rl_control import Controller
from visualization import plot_all_scenarios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--load', action='store_true', help='не обучать, загрузить сохранённые веса')
    ap.add_argument('--iters', type=int, default=None, help='итераций PPO')
    args = ap.parse_args()

    print("=" * 70)
    print("АПРОБАЦИЯ КОНТУРА АВТОМАТИЧЕСКОГО УПРАВЛЕНИЯ (RL-регулятор)")
    print("=" * 70)
    hw.configure_torch_threads()
    print(hw.describe())

    if args.load:
        feat, ac = load()
    else:
        with hw.keep_awake():
            ac = train_rl(iters=args.iters, validate=True)
        feat = RL['FEAT']
        print(f"Сохранено: {save(ac)}")
    controller = Controller(ac, feat)

    results_df, false_alarm_rate = run_full_experiment(controller)

    print("\n" + "=" * 70)
    print("Таблица 1 - Результаты апробации (среднее по 30 реализациям)")
    print("=" * 70)
    print(results_df.to_string(index=False))
    print(f"\nS4: доля тактов с ложным срабатыванием регулятора: {false_alarm_rate:.2f}%")

    plot_all_scenarios(controller)

    print("\nЭксперимент завершён успешно!")


if __name__ == '__main__':
    main()