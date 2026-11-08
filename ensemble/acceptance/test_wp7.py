import os

import pytest
import numpy as np

WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'controller_rl.pt')

def test_soft_deadzone():
    """Проверяет, что мягкая мёртвая зона обнуляет малые сигналы."""
    import torch
    from rl_core import soft_deadzone

    thr = 0.1
    # малые сигналы (в том числе ровно на пороге) -> строго ноль
    small = torch.tensor([0.0, 0.05, -0.05, 0.1, -0.1, 0.099])
    out_small = soft_deadzone(small, thr)
    assert torch.all(out_small == 0), "Сигналы |u| <= порога должны обнуляться"

    # большие сигналы не обнуляются и сдвинуты к нулю на порог
    big = torch.tensor([0.5, -0.5, 1.0])
    out_big = soft_deadzone(big, thr)
    assert torch.all(out_big != 0), "Сигналы |u| > порога не должны обнуляться"
    assert torch.allclose(out_big, torch.sign(big) * (big.abs() - thr)), \
        "Сигнал выше порога должен быть сдвинут к нулю на величину порога"

def test_improved_reward():
    """Проверяет, что веса награды были изменены."""
    from config_rl import RL
    assert RL['REWARD']['K_E'] >= 2.0, "Вес K_E должен быть увеличен"
    assert RL['REWARD']['K_ZONE'] >= 2.0, "Вес K_ZONE должен быть увеличен"


def test_training_iterations():
    """Проверяет, что обучение PPO настроено на 1000 итераций (требование 3)."""
    from config_rl import RL
    assert RL['PPO']['ITERS'] >= 1000, "Число итераций PPO должно быть не меньше 1000"


@pytest.mark.skipif(not os.path.exists(WEIGHTS), reason="нет обученных весов controller_rl.pt")
def test_strict_zero_on_unexcited_channels():
    """Выход регулятора строго нулевой на невозмущённых каналах (требование 1)."""
    import torch
    import rl_control
    import rl_core as R
    from config import G, N_PARAMS, T_TOTAL
    from scenarios import get_disturbance_profile

    ctrl = rl_control.load_controller()
    policy = ctrl.policy()
    nominal = R._plant(np.random.RandomState(0), R.RL['NOMINAL'])
    rate, jump = get_disturbance_profile('S2', 50, 1.0)         # скачок только по x6
    item = dict(nominal, rate=rate, jump=jump, t0=50, quiet=False,
                noise=np.zeros((T_TOTAL - 1, N_PARAMS)),        # без шума: остальные каналы не возмущены
                cref=R._cref_eval(), name='')
    env = R.RLEnv(torch.device('cpu'), reward=False, pid_inner=False)
    env.reset(R.to_device(R._pack([item]), torch.device('cpu')))
    policy.reset(env)

    U = np.zeros((T_TOTAL, N_PARAMS))
    N = np.zeros((T_TOTAL, N_PARAMS))
    with torch.no_grad():
        for t in range(T_TOTAL):
            u = policy.act(env)
            U[t] = u.numpy()
            N[t] = env.N.numpy()[0, t]
            if t < R.N_STEPS:
                env.step(u)

    dev = np.abs(N - G)
    quiet = [q for q in range(N_PARAMS) if dev[:, q].max() == 0.0]
    assert quiet, "в выборке нет каналов с нулевым отклонением"
    for q in quiet:
        assert np.all(U[:, q] == 0.0), f"Канал x{q + 1} не отклонялся, но выход регулятора не нулевой"
    disturbed = [q for q in range(N_PARAMS) if dev[:, q].max() > 0]
    assert any(np.abs(U[:, q]).max() > 0 for q in disturbed), \
        "возмущённый канал не получил ни одной команды - регулятор не работает"
