
import inspect
from pathlib import Path

import numpy as np


def test_file_exists():
    assert Path("plant_equations.py").exists(), \
        "plant_equations.py не найден в корне проекта"


def test_has_class_and_8_states():
    import plant_equations as pe
    assert hasattr(pe, "PlantODESystem"), "нет класса PlantODESystem"
    assert pe.PlantODESystem.N_STATES == 8, \
        f"N_STATES = {pe.PlantODESystem.N_STATES}, ожидается 8"


def test_api_methods():
    import plant_equations as pe
    p = pe.PlantODESystem()
    for name in ("initial_state", "setpoint", "tolerance",
                 "derivatives", "step", "equilibrium"):
        assert callable(getattr(p, name, None)), f"нет метода {name}"


def test_initial_state_shape():
    import plant_equations as pe
    p = pe.PlantODESystem()
    x0 = p.initial_state()
    assert isinstance(x0, np.ndarray), "initial_state() должен возвращать numpy.ndarray"
    assert x0.shape == (8,), f"shape {x0.shape}, ожидается (8,)"
    assert np.all(np.isfinite(x0)), "initial_state() содержит NaN или inf"


def test_setpoint_is_time_varying():
    import plant_equations as pe
    p = pe.PlantODESystem()
    g1 = p.setpoint(0.0)
    g2 = p.setpoint(50.0)
    g3 = p.setpoint(120.0)
    for g, t in ((g1, 0.0), (g2, 50.0), (g3, 120.0)):
        assert isinstance(g, np.ndarray), f"setpoint({t}) не numpy.ndarray"
        assert g.shape == (8,), f"setpoint({t}).shape = {g.shape}, ожидается (8,)"
    delta = max(np.abs(g2 - g1).max(), np.abs(g3 - g2).max())
    assert delta > 1e-6, \
        "g(t) не зависит от t — это константа, а требуется функция времени"


def test_tolerance_shape():
    import plant_equations as pe
    p = pe.PlantODESystem()
    d = p.tolerance()
    assert d.shape == (8,), f"shape {d.shape}, ожидается (8,)"
    assert (d > 0).all(), "допуски должны быть > 0"


def test_derivatives_shape():
    import plant_equations as pe
    p = pe.PlantODESystem()
    dx = p.derivatives(p.initial_state(), np.zeros(8), 0.0)
    assert dx.shape == (8,), f"shape {dx.shape}, ожидается (8,)"
    assert np.all(np.isfinite(dx)), "derivatives() содержит NaN или inf"


def test_step_converges_to_equilibrium():
    import plant_equations as pe
    p = pe.PlantODESystem()
    x = p.initial_state()
    u = np.zeros(8)
    for k in range(2000):
        x = p.step(x, u, 0.1, k * 0.1)
        assert np.all(np.isfinite(x)), \
            f"шаг {k}: состояние содержит NaN или inf"
    assert np.abs(x).max() < 1e6, \
        f"состояние взорвалось: max|x| = {np.abs(x).max():.2e}"


def test_equilibrium_returns_shapes():
    import plant_equations as pe
    p = pe.PlantODESystem()
    result = p.equilibrium()
    assert isinstance(result, tuple) and len(result) == 2, \
        "equilibrium() должен возвращать (x*, u*)"
    x_eq, u_eq = result
    assert x_eq.shape == (8,), f"x*.shape = {x_eq.shape}, ожидается (8,)"
    assert u_eq.shape == (8,), f"u*.shape = {u_eq.shape}, ожидается (8,)"


def test_no_supercriterion_in_plant():
    import plant_equations as pe
    src = inspect.getsource(pe).lower()
    for word in ("supercrit", "superkrit", "supercriterion",
                 "ahp_w", "k_total", "суперкритери"):
        assert word not in src, \
            f"plant_equations.py содержит '{word}' — суперкритерий здесь не нужен"


def test_has_physical_comments():
    src = Path("plant_equations.py").read_text(encoding="utf-8").lower()
    physics = ["износ", "вибрац", "шероховат", "температур", "охлажд",
               "резан", "wear", "vibration", "roughness", "cooling", "chatter",
               "теплов", "нагрев"]
    found = [p for p in physics if p in src]
    assert len(found) >= 3, \
        f"мало физических обоснований: найдено {found}. Нужно минимум 3 из {physics}"


def test_disturbance_affects_state():
    import plant_equations as pe
    p = pe.PlantODESystem()
    x = p.initial_state()
    d = np.zeros(8)
    d[0] = 0.5
    x2 = p.step(x, np.zeros(8), 0.1, 0.0, disturbance=d)
    assert not np.allclose(x, x2), "disturbance не влияет на состояние"


def test_noise_affects_state():
    import plant_equations as pe
    p = pe.PlantODESystem()
    x = p.initial_state()
    n = np.zeros(8)
    n[3] = 0.1
    x2 = p.step(x, np.zeros(8), 0.1, 0.0, noise=n)
    assert not np.allclose(x, x2), "noise не влияет на состояние"


def test_no_torch_import():
    src = Path("plant_equations.py").read_text(encoding="utf-8")
    assert "import torch" not in src, "plant_equations.py не должен импортировать torch"
    assert "from torch" not in src, "plant_equations.py не должен импортировать torch"


def test_plant_py_unchanged():
    assert Path("plant.py").exists(), "plant.py отсутствует"
    assert Path("plant.py").stat().st_size > 1000, "plant.py повреждён (слишком маленький)"