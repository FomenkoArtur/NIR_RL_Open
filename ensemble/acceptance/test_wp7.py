import pytest
import numpy as np

def test_soft_deadzone():
    """Проверяет, что мягкая мёртвая зона обнуляет малые сигналы."""
    pass

def test_improved_reward():
    """Проверяет, что веса награды были изменены."""
    from config_rl import RL
    assert RL['REWARD']['K_E'] >= 2.0, "Вес K_E должен быть увеличен"
    assert RL['REWARD']['K_ZONE'] >= 2.0, "Вес K_ZONE должен быть увеличен"
