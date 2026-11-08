"""
Приёмочные тесты для WP6 — GUI.
"""
import re
from pathlib import Path


def test_no_without_controller():
    src = Path("gui/main_window.py").read_text(encoding="utf-8")
    assert "no_ctrl_check" not in src, "чекбокс «Без регулятора» не убран"
    assert 'MODE_LABELS[\'none\']' not in src


def test_nomenclature_present():
    src = Path("gui/main_window.py").read_text(encoding="utf-8")
    assert "номенклатур" in src.lower() or "nomenclature" in src.lower()
    assert Path("gui/nomenclature.py").exists()


def test_three_charts():
    src = Path("gui/plots.py").read_text(encoding="utf-8")
    for name in ("PID", "MPC"):
        assert name in src, f"нет графика для {name}"
    assert "RL" in src or "cnn" in src.lower() or "CNN" in src


def test_afc_tab():
    src = Path("gui/main_window.py").read_text(encoding="utf-8")
    assert "АФХ" in src or "afc" in src.lower()
    assert Path("gui/afc_bode.py").exists()


def test_bode_tab():
    src = Path("gui/main_window.py").read_text(encoding="utf-8")
    assert "Боде" in src or "bode" in src.lower()


def test_ahp_only_in_visualization():
    training_files = ["rl_core.py", "rl_trainer.py", "rl_control.py", "episode.py", "trainer.py", "experiment.py"]
    for f in training_files:
        p = Path(f)
        if not p.exists(): continue
        src = p.read_text(encoding="utf-8")
        assert not re.search(r"ahp|МАИ|Saaty|supercrit", src, re.I), f"{f}: МАИ в обучении"


def test_ahp_weights_module():
    assert Path("gui/ahp_weights.py").exists()
    import gui.ahp_weights as a
    funcs = [n for n in dir(a) if not n.startswith("_")]
    assert any("weight" in f.lower() or "ahp" in f.lower() for f in funcs)


def test_time_varying_weights():
    src = Path("gui/ahp_weights.py").read_text(encoding="utf-8")
    assert re.search(r"def \w+\([^)]*t[^)]*\)", src), "нет функции, зависящей от времени"
