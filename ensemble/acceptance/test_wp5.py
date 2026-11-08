"""
Приёмочные тесты для WP5 — PID и MPC на PlantODESystem.
"""
from pathlib import Path

import numpy as np


def test_pid_module():
    assert Path("pid_control.py").exists(), "нет pid_control.py"
    import pid_control
    assert hasattr(pid_control, "PIDController"), "нет класса PIDController"


def test_mpc_module():
    assert Path("mpc_control.py").exists(), "нет mpc_control.py"
    import mpc_control
    assert hasattr(mpc_control, "MPCController"), "нет класса MPCController"


def test_pid_api():
    import pid_control
    from plant_equations import PlantODESystem
    p = PlantODESystem()
    pid = pid_control.PIDController(p)
    pid.reset()
    u = pid.act(p.initial_state(), 0.0)
    assert isinstance(u, np.ndarray)
    assert u.shape == (8,)
    assert np.all(np.isfinite(u))


def test_mpc_api():
    import mpc_control
    from plant_equations import PlantODESystem
    p = PlantODESystem()
    mpc = mpc_control.MPCController(p)
    mpc.reset()
    u = mpc.act(p.initial_state(), 0.0)
    assert isinstance(u, np.ndarray)
    assert u.shape == (8,)
    assert np.all(np.isfinite(u))


def test_pid_saturation():
    import pid_control
    from plant_equations import PlantODESystem
    p = PlantODESystem()
    pid = pid_control.PIDController(p, u_max=0.5)
    pid.reset()
    x_far = p.initial_state() + 100.0
    u = pid.act(x_far, 0.0)
    assert np.abs(u).max() <= 0.5 + 1e-6


def test_mpc_saturation():
    import mpc_control
    from plant_equations import PlantODESystem
    p = PlantODESystem()
    mpc = mpc_control.MPCController(p, u_max=0.5)
    mpc.reset()
    x_far = p.initial_state() + 100.0
    u = mpc.act(x_far, 0.0)
    assert np.abs(u).max() <= 0.5 + 1e-6


def test_pid_reduces_error():
    import pid_control
    from plant_equations import PlantODESystem
    p = PlantODESystem()
    pid = pid_control.PIDController(p, kp=2.0, ki=0.1, kd=0.05)
    pid.reset()
    x = p.initial_state() + 5.0
    g = p.setpoint(0.0)
    e0 = np.abs(x - g).mean()
    for step in range(50):
        u = pid.act(x, step * 0.1)
        x = p.step(x, u, 0.1, step * 0.1)
    e1 = np.abs(x - p.setpoint(50 * 0.1)).mean()
    assert e1 < e0, f"PID не уменьшает ошибку: e0 = {e0:.3f}, e1 = {e1:.3f}"


def test_mpc_reduces_error():
    import mpc_control
    from plant_equations import PlantODESystem
    p = PlantODESystem()
    mpc = mpc_control.MPCController(p, horizon=5, Q=1.0, R=0.1)
    mpc.reset()
    x = p.initial_state() + 5.0
    g = p.setpoint(0.0)
    e0 = np.abs(x - g).mean()
    for step in range(50):
        u = mpc.act(x, step * 0.1)
        x = p.step(x, u, 0.1, step * 0.1)
    e1 = np.abs(x - p.setpoint(50 * 0.1)).mean()
    assert e1 < e0, f"MPC не уменьшает ошибку: e0 = {e0:.3f}, e1 = {e1:.3f}"


def test_no_torch_in_controllers():
    for f in ("pid_control.py", "mpc_control.py"):
        src = Path(f).read_text(encoding="utf-8")
        assert "import torch" not in src
        assert "from torch" not in src


def test_physical_comments():
    for f in ("pid_control.py", "mpc_control.py"):
        src = Path(f).read_text(encoding="utf-8").lower()
        terms = ["пропорц", "интеграл", "дифференц", "насыщ", "горизонт",
                 "прогноз", "ошибк", "коэффициент", "управлен"]
        found = sum(1 for t in terms if t in src)
        assert found >= 2, f"{f}: мало физических комментариев"


def test_plant_equations_unchanged():
    assert Path("plant_equations.py").exists()
    assert Path("plant_equations.py").stat().st_size > 1000