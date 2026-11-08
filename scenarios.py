"""
Сценарии особых причин изменчивости (п. 3.2.1).
"""

import numpy as np
from config import A_CUR, CRITICAL_SCENARIOS, N_PARAMS, T_TOTAL, T_START, SCENARIOS
from plant import plant


# ========== ФИЗИЧНЫЕ ТРЕНДЫ ТЕСТОВОЙ ВЫБОРКИ (T1-T3) ==========
# Форма каждого тренда - хевисайд x линейная функция:
#     rate(t)[i] = v * 1[t >= t0],
# то есть до момента t0 возмущение отсутствует, а после t0 к состоянию каждый
# такт добавляется постоянная скорость v, поэтому накопленное отклонение растёт
# линейно (та же постоянная скорость, что в сценарии S1). Амплитуды v заданы в
# долях размаха контрольного коридора S_i и пересчитываются в абсолютные единицы
# умножением на plant.S - как в get_disturbance_profile и custom_profile.
_PHYSICAL_TRENDS = [
    {
        'name': 'T1',
        'param': 0,
        'rate': 0.027,
        't0': 60,
        'physical': 'тепловой дрейф шпинделя',
        # T1 - тепловой дрейф шпинделя (токарная операция): при длительной работе шпиндель, передняя
        # бабка и станина нагреваются, тепловое расширение смещает ось вращения, и диаметр обработанной
        # поверхности x1 уводится от настройки. После прогрева t0 = 60 тактов тепловой режим становится
        # стационарным, поэтому увод диаметра нарастает с постоянной скоростью 0.027 S за такт - это
        # медленный систематический тренд, который надо успеть скомпенсировать до выхода за допуск.
    },
    {
        'name': 'T2',
        'param': 5,
        'rate': 0.020,
        't0': 80,
        'physical': 'усталостная вибрация оправки фрезы',
        # T2 - усталостная вибрация (фрезерная операция): накопление усталостных микротрещин в оправке
        # фрезы и в посадочном конусе снижает её жёсткость, из-за чего растёт амплитуда вибрации x6.
        # После t0 = 80 тактов наработки дефект развивается лавинообразно, и вибрация растёт линейно
        # со скоростью 0.020 S за такт - типичная картина усталостного разрушения инструмента.
    },
    {
        'name': 'T3',
        'param': 7,
        'rate': -0.015,
        't0': 40,
        'physical': 'загрязнение теплообменника → деградация охлаждения',
        # T3 - деградация охлаждения (закалка): загрязнение теплообменника и зарастание каналов накипью
        # снижают теплосъём, поэтому скорость охлаждения x8 падает. После t0 = 40 тактов загрязнение
        # становится заметным, и скорость охлаждения линейно уменьшается на 0.015 S за такт; заготовка
        # выходит из ванны перегретой, и твёрдость закалённого слоя снижается.
    },
]

_PHYSICAL_TRENDS_BY_NAME = {trend['name']: trend for trend in _PHYSICAL_TRENDS}


def get_physical_trends():
    """
    Возвращает список физичных трендов тестовой выборки:
    [
        {'name': 'T1', 'param': 0, 'rate': 0.027, 't0': 60,
         'physical': 'тепловой дрейф шпинделя'},
        {'name': 'T2', 'param': 5, 'rate': 0.020, 't0': 80,
         'physical': 'усталостная вибрация оправки фрезы'},
        {'name': 'T3', 'param': 7, 'rate': -0.015, 't0': 40,
         'physical': 'загрязнение теплообменника → деградация охлаждения'},
    ]

    rate - полная амплитуда скорости в долях размаха S_i, t0 - момент начала
    тренда в тактах. Копии словарей защищают таблицу от случайной правки.
    """
    return [dict(trend) for trend in _PHYSICAL_TRENDS]


def get_physical_trend_profile(name):
    """
    Возвращает (rate, jump) для тренда T1-T3 - в том же формате, что
    get_disturbance_profile и custom_profile:
    rate - добавка к состоянию на каждом такте, jump - скачок на такте t0
    (у физичных трендов скачка нет).

    Форма - хевисайд x линейная функция: rate[t, i] = v * 1[t >= t0],
    то есть до t0 возмущения нет, после t0 скорость постоянна и отклонение
    растёт линейно. Амплитуда пересчитана в абсолютные единицы умножением
    на plant.S.
    """
    spec = _PHYSICAL_TRENDS_BY_NAME.get(name)
    if spec is None:
        known = ', '.join(trend['name'] for trend in _PHYSICAL_TRENDS)
        raise ValueError(f'неизвестный физичный тренд {name!r}, известны: {known}')

    rate = np.zeros((T_TOTAL, N_PARAMS))
    jump = np.zeros(N_PARAMS)
    t0 = int(spec['t0'])
    assert 0 <= t0 < T_TOTAL, f'{name}: t0={t0} за пределами эпизода (T_TOTAL={T_TOTAL})'

    rate[t0:, spec['param']] = spec['rate']      # хевисайд 1[t >= t0] * постоянная скорость
    return rate * plant.S, jump * plant.S


def get_disturbance_profile(scenario, t0=T_START, scale=1.0):
    """
    Формирование профиля возмущений для заданного сценария.
    """
    rate = np.zeros((T_TOTAL, N_PARAMS))
    jump = np.zeros(N_PARAMS)

    # S4 - контрольный сценарий без особой причины
    if scenario == 'S4':
        return rate * plant.S, jump * plant.S

    # T1-T3 - физичные тренды тестовой выборки; момент начала t0 задан физикой
    # процесса (тепловой дрейф, усталостная вибрация, загрязнение теплообменника),
    # поэтому аргумент t0 для них не используется, а scale общий с S-сценариями.
    if scenario in _PHYSICAL_TRENDS_BY_NAME:
        rate, jump = get_physical_trend_profile(scenario)
        return rate * scale, jump * scale

    config = SCENARIOS[scenario]

    if config['rate_param'] is not None:
        rate[t0:, config['rate_param']] = config['rate_value'] * scale

    if config['jump_param'] is not None:
        jump[config['jump_param']] = config['jump_value'] * scale

    return rate * plant.S, jump * plant.S


def custom_profile(param, kind, value, t0=T_START):
    """
    Профиль возмущения на произвольный показатель (для обучающей выборки).
    kind: 'trend' - линейный тренд, 'jump' - скачок.
    """
    rate = np.zeros((T_TOTAL, N_PARAMS))
    jump = np.zeros(N_PARAMS)

    if kind == 'trend':
        rate[t0:, param] = value
    else:
        jump[param] = value

    return rate * plant.S, jump * plant.S


def critical_profile(name):
    """
    Профиль возмущений критической ситуации тестовой выборки (CRITICAL_SCENARIOS в config.py).
    Возвращает (rate, jump) в том же формате, что get_disturbance_profile и custom_profile:
    rate - добавка к состоянию на каждом такте, jump - скачок на такте t0.
    Амплитуды в конфиге заданы в долях размаха S_i, здесь пересчитываются умножением на plant.S.
    """
    spec = CRITICAL_SCENARIOS[name]
    t0 = spec['t0']
    rate = np.zeros((T_TOTAL, N_PARAMS))
    jump = np.zeros(N_PARAMS)

    for comp in spec['components']:
        kind, p, value = comp['kind'], comp['param'], comp['value']
        start = t0 + comp.get('start', 0)
        assert start < T_TOTAL, f'{name}: компонента начинается за пределами эпизода'

        if kind == 'jump':
            if start == t0:
                jump[p] += value                    # как скачок обучающей выборки
            else:
                rate[start, p] += value             # скачок позже t0 (каскад)
        elif kind == 'trend':
            rate[start:, p] += value
        elif kind == 'pulse':
            end = start + comp['duration']
            assert end < T_TOTAL, f'{name}: импульс выходит за пределы эпизода'
            rate[start, p] += value
            rate[end, p] -= value * A_CUR ** comp['duration']   # возврат с учётом инерции объекта
        elif kind == 'sine':
            period = comp['period']
            k = np.arange(T_TOTAL - start)
            # колебания состояния с амплитудой value: производная от value * sin(2*pi*k/period)
            rate[start:, p] += value * (2 * np.pi / period) * np.cos(2 * np.pi * k / period)
        else:
            raise ValueError(f'{name}: неизвестный вид компоненты {kind!r}')

    return rate * plant.S, jump * plant.S


def get_scenario_description(scenario):
    """Получение текстового описания сценария."""
    if scenario == 'S4':
        return 'Без особой причины (контрольный сценарий)'
    if scenario in _PHYSICAL_TRENDS_BY_NAME:
        return _PHYSICAL_TRENDS_BY_NAME[scenario]['physical']
    return SCENARIOS[scenario]['description']