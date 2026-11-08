import numpy as np
from pathlib import Path

def test_file_exists():
    assert Path("plant_equations.py").exists()

def test_has_plant_class():
    import plant_equations as pe
    assert hasattr(pe, "PlantODESystem")
    assert pe.PlantODESystem.N_STATES == 2

def test_initial_state_shape():
    import plant_equations as pe
    p = pe.PlantODESystem()
    assert p.initial_state().shape == (2,)

def test_derivatives_shape():
    import plant_equations as pe
    p = pe.PlantODESystem()
    dx = p.derivatives(p.initial_state(), np.zeros(2), 0.0)
    assert dx.shape == (2,)

def test_step_no_nan():
    import plant_equations as pe
    p = pe.PlantODESystem()
    x = p.initial_state()
    for _ in range(10):
        x = p.step(x, np.zeros(2), 0.1, 0.0)
    assert not np.isnan(x).any()