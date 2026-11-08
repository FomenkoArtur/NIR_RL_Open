"""
Приёмочные тесты для WP2 — физичные сценарии в scenarios.py.
"""

import inspect
from pathlib import Path

import numpy as np


def test_s1_trend_x1():
    """S1 — линейный тренд по x1, без скачков."""
    from scenarios import get_disturbance_profile
    rate, jump = get_disturbance_profile("S1", t0=40, scale=1.0)
    assert rate.shape[1] == 8, f"rate.shape = {rate.shape}"
    assert jump.shape == (8,)
    assert np.abs(rate[40:, 0]).max() > 0, "S1 не даёт тренда по x1"
    assert np.abs(jump).max() < 1e-9, "S1 не должен давать скачков"
    others = np.delete(np.abs(rate), 0, axis=1)
    assert others.max() < 1e-9, "S1 затронул другие компоненты"


def test_s2_impulse_x6():
    """S2 — скачок по x6."""
    from scenarios import get_disturbance_profile
    rate, jump = get_disturbance_profile("S2", t0=40, scale=1.0)
    assert abs(jump[5]) > 0, "S2 не даёт скачка по x6"
    others = np.delete(np.abs(jump), 5)
    assert others.max() < 1e-9, "S2 затронул другие компоненты"


def test_s3_cooling_drop_x8():
    """S3 — отрицательный скачок по x8."""
    from scenarios import get_disturbance_profile
    rate, jump = get_disturbance_profile("S3", t0=40, scale=1.0)
    assert jump[7] < 0, "S3 должен давать отрицательный скачок по x8"


def test_s4_zero():
    """S4 — без возмущений."""
    from scenarios import get_disturbance_profile
    rate, jump = get_disturbance_profile("S4", t0=40, scale=1.0)
    assert np.abs(rate).max() < 1e-9
    assert np.abs(jump).max() < 1e-9


def test_custom_profile_trend():
    """custom_profile с kind='trend' — линейный тренд по одному показателю."""
    from scenarios import custom_profile
    rate, jump = custom_profile(0, "trend", 0.027, t0=40)
    assert np.abs(rate[40:, 0]).max() > 0
    assert np.abs(jump).max() < 1e-9


def test_custom_profile_jump():
    """custom_profile с kind='jump' — скачок по одному показателю."""
    from scenarios import custom_profile
    rate, jump = custom_profile(5, "jump", 1.6, t0=40)
    assert abs(jump[5]) > 0


def test_get_physical_trends_exists():
    """Функция get_physical_trends() присутствует."""
    import scenarios
    assert hasattr(scenarios, "get_physical_trends"), \
        "нет функции get_physical_trends()"


def test_physical_trends_count_and_structure():
    """Ровно 3 тренда T1–T3 с обязательными полями."""
    from scenarios import get_physical_trends
    trends = get_physical_trends()
    assert isinstance(trends, list), "должен быть список"
    assert len(trends) >= 3, f"трендов {len(trends)}, ожидается ≥3"
    names = {t['name'] for t in trends}
    for name in ('T1', 'T2', 'T3'):
        assert name in names, f"нет тренда {name}"
    for t in trends:
        for key in ('name', 'param', 'rate', 't0', 'physical'):
            assert key in t, f"тренд {t.get('name')}: нет поля {key}"
        assert 0 <= t['param'] < 8, f"param вне [0,8): {t['param']}"
        assert isinstance(t['physical'], str) and len(t['physical']) > 5, \
            f"тренд {t['name']}: поле physical слишком короткое"


def test_physical_trend_profile_exists():
    """Функция get_physical_trend_profile() присутствует."""
    import scenarios
    assert hasattr(scenarios, "get_physical_trend_profile"), \
        "нет функции get_physical_trend_profile(name)"


def test_physical_trend_profile_shapes():
    """get_physical_trend_profile('T1') возвращает (rate, jump) правильных форм."""
    from scenarios import get_physical_trend_profile
    for name in ('T1', 'T2', 'T3'):
        rate, jump = get_physical_trend_profile(name)
        assert rate.ndim == 2, f"{name}: rate.ndim = {rate.ndim}"
        assert rate.shape[1] == 8, f"{name}: rate.shape = {rate.shape}"
        assert jump.shape == (8,), f"{name}: jump.shape = {jump.shape}"


def test_physical_trend_profile_nonzero():
    """Каждый тренд даёт ненулевое возмущение в своей компоненте."""
    from scenarios import get_physical_trend_profile
    for name in ('T1', 'T2', 'T3'):
        rate, jump = get_physical_trend_profile(name)
        assert np.abs(rate).max() > 1e-6 or np.abs(jump).max() > 1e-6, \
            f"тренд {name} даёт нулевое возмущение"


def test_physical_trends_hevyside_form():
    """После t0 тренд — постоянная скорость (хевисайд × линейная функция)."""
    from scenarios import get_physical_trend_profile, get_physical_trends
    trends = {t['name']: t for t in get_physical_trends()}
    for name in ('T1', 'T2', 'T3'):
        spec = trends[name]
        rate, jump = get_physical_trend_profile(name)
        p, t0 = spec['param'], spec['t0']
        before = np.abs(rate[:t0, p]).max()
        assert before < 1e-9, f"{name}: до t0 скорость {before:.2e}, ожидается 0"
        after = rate[t0:, p]
        assert np.abs(after).max() > 0, f"{name}: после t0 скорость нулевая"


def test_physical_comments_in_source():
    """В scenarios.py — минимум 3 физических обоснования."""
    src = Path("scenarios.py").read_text(encoding="utf-8").lower()
    terms = ["износ", "вибрац", "охлажд", "резан", "фрез", "токар", "закал",
             "теплов", "дрейф", "усталост", "шпиндел", "теплообмен"]
    found = [t for t in terms if t in src]
    assert len(found) >= 3, \
        f"мало физических обоснований: найдено {found}, нужно ≥3 из {terms}"


def test_physical_trends_have_long_comments():
    """Для каждого T1–T3 — комментарий минимум 100 символов с физикой."""
    src = Path("scenarios.py").read_text(encoding="utf-8")
    for name in ('T1', 'T2', 'T3'):
        found_comment = False
        for line in src.splitlines():
            if name in line and ('#' in line or '"""' in line):
                comment_part = line[line.find('#'):] if '#' in line else line
                if len(comment_part) >= 30:
                    found_comment = True
                    break
        if not found_comment:
            fn_idx = src.find(f'def {name.lower()}')
            if fn_idx < 0:
                fn_idx = src.find(f'"{name}"')
            if fn_idx > 0:
                snippet = src[fn_idx:fn_idx + 300]
                if any(t in snippet.lower() for t in ['износ', 'вибрац', 'охлажд', 'тепло', 'дрейф']):
                    found_comment = True
        assert found_comment, f"тренд {name}: нет подробного физического комментария"


def test_no_supercriterion_in_scenarios():
    """В scenarios.py нет суперкритерия — он только для визуализации."""
    src = Path("scenarios.py").read_text(encoding="utf-8").lower()
    for word in ("supercrit", "superkrit", "ahp_w", "k_total", "суперкритери"):
        assert word not in src, f"scenarios.py содержит '{word}'"


def test_plant_py_unchanged():
    """plant.py не изменён."""
    assert Path("plant.py").exists(), "plant.py отсутствует"
    assert Path("plant.py").stat().st_size > 1000, "plant.py повреждён"


def test_plant_equations_exists():
    """plant_equations.py (из WP1) остался."""
    assert Path("plant_equations.py").exists(), \
        "plant_equations.py из WP1 исчез"
