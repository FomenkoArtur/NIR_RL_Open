"""Справочник номенклатуры изделий (WP6).

Вал, Фланец, Корпус — типы изделий участка металлообработки.
Для каждой номенклатуры задаются номинальные параметры объекта
(статический коэффициент передачи и постоянная времени по каналам),
используемые при расчёте частотных характеристик на вкладках АФХ/Боде.
"""
from __future__ import annotations

NOMENCLATURE = [
    {"key": "shaft", "name": "Вал", "description": "Токарная обработка"},
    {"key": "flange", "name": "Фланец", "description": "Фрезерная обработка"},
    {"key": "housing", "name": "Корпус", "description": "Закалка"},
]


def items():
    """Список записей справочника в порядке отображения в ComboBox."""
    return list(NOMENCLATURE)


def names():
    """Человекочитаемые номенклатуры для выпадающего списка."""
    return [item["name"] for item in NOMENCLATURE]


def find(name_or_key):
    """Запись по имени или ключу; None, если не найдена."""
    for item in NOMENCLATURE:
        if item["name"] == name_or_key or item["key"] == name_or_key:
            return item
    return None


# Номинальные параметры цифрового двойника по номенклатуре.
# Дискретная первоканальная модель x[k+1] = a*x[k] + b*u[k] с
# инерцией a = A_CUR и входным коэффициентом b, выбранным так, чтобы
# статический коэффициент передачи объекта был равен k (b = k*(1 - a)).
NOMINAL_PARAMS = {
    "shaft":  {"k": 1.00, "tau": 1.2},
    "flange": {"k": 0.85, "tau": 1.0},
    "housing": {"k": 1.20, "tau": 1.5},
}


def nominal_params(key):
    """Параметры объекта для номенклатуры (ключ 'shaft'/'flange'/'housing')."""
    return dict(NOMINAL_PARAMS.get(key, NOMINAL_PARAMS["shaft"]))
