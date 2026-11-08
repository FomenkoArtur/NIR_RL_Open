@'
"""
Приёмочные тесты для WP3 — структура CNN 8->8, без обучения.
"""
import re
from pathlib import Path

import numpy as np


def test_n_features_is_8():
    import config
    assert config.N_FEATURES == 8, f"N_FEATURES = {config.N_FEATURES}, ожидается 8"


def test_n_params_is_8():
    import config
    assert config.N_PARAMS == 8


def test_no_supercriterion_in_training_files():
    files = ["rl_core.py", "rl_trainer.py", "rl_control.py", "episode.py", "trainer.py"]
    pattern = re.compile(r"supercrit|superkrit|supercriterion|ahp_w\b|\bK_total\b", re.I)
    for f in files:
        p = Path(f)
        if not p.exists():
            continue
        m = pattern.search(p.read_text(encoding="utf-8"))
        assert m is None, f"{f} содержит {m.group(0)}"


def test_tune_cnn_no_architecture_search():
    p = Path("tune_cnn.py")
    if not p.exists():
        return
    src = p.read_text(encoding="utf-8")
    assert "class TuneCNN" not in src, "класс TuneCNN должен быть удалён"
    assert "suggest_rl" not in src, "suggest_rl должен быть удалён"


def test_actor_output_is_8():
    import config
    from rl_core import ActorCritic, RL_ARCH
    hp = dict(RL_ARCH)
    ac = ActorCritic(hp)
    assert ac.actor.fc2.out_features == config.N_PARAMS


def test_window_shape_is_8():
    import torch
    import config
    from rl_core import RLEnv, to_device, sample_episodes
    dev = torch.device("cpu")
    rng = np.random.RandomState(0)
    ep = to_device(sample_episodes(rng, 2), dev)
    env = RLEnv(dev, reward=False)
    env.reset(ep)
    w = env.window("u")
    assert w.shape[0] == 2
    assert w.shape[1] == config.N_FEATURES, f"каналов {w.shape[1]}, ожидается {config.N_FEATURES}"
    assert w.shape[2] == config.WINDOW


def test_no_w32_in_window():
    import inspect
    from rl_core import RLEnv
    src = inspect.getsource(RLEnv.window)
    assert "w32" not in src, "признак w_i*N_i (w32) должен быть убран из RLEnv.window"


def test_plant_py_unchanged():
    assert Path("plant.py").exists()
    assert Path("plant.py").stat().st_size > 1000


def test_plant_equations_exists():
    assert Path("plant_equations.py").exists()
'@ | Set-Content -Encoding utf8 ensemble\acceptance\test_wp3.py