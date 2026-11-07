"""
Конфигурация эксперимента.
"""

import json
import os

import numpy as np

_TUNED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tuned')


def _load(name):
    path = os.path.join(_TUNED_DIR, name)
    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return {}


# ========== ВЕСА ==========
N_PARAMS = 8
W_OPS = np.array([0.5, 0.3, 0.2])
N_GRP = np.array([3, 3, 2])

# ========== НОРМАЛИЗАЦИЯ ==========
BETA = 100.0
N_WINDOW = 200
GAMMA = 1.0
S_MIN_FRAC = 1 / 6
IQR_TO_SIGMA = 1.349
Z_LIMIT = 3.0
N_LOW = 1 / (1 + np.exp(Z_LIMIT))
N_HIGH = 1 / (1 + np.exp(-Z_LIMIT))

# ========== ЗОНЫ УПРАВЛЕНИЯ ==========
ZONE_LAYOUTS = {
    'single': [list(range(N_PARAMS))],
    'operation': [[0, 1, 2], [3, 4, 5], [6, 7]],
    'parameter': [[i] for i in range(N_PARAMS)],
}
CONTROL_LAYOUT = 'parameter'
ZONES = ZONE_LAYOUTS[CONTROL_LAYOUT]
N_ZONES = len(ZONES)
N_FEATURES = 1 + 2 * N_PARAMS

# ========== ВРЕМЕННЫЕ ПАРАМЕТРЫ ==========
WINDOW = 32
PID_WINDOW = 10
DELTA = 0.03
N_CONS = 9
T_OBS = 100
T_START = 40
N_REAL = 30
T_TOTAL = T_START + T_OBS
N_BURN_IN = N_WINDOW
assert T_START >= WINDOW
G = 0.5

# ========== ПАРАМЕТРЫ ЦИФРОВОГО ДВОЙНИКА ==========
A_CUR = 0.99
SIG_CUR = 0.0067
A_BASE = 0.90
SIG_BASE = 0.30
N_CALIB = 200
A2 = 2.66
D4 = 3.267
D3 = 0.0
C_RAW_FACTOR = 0.10

# ========== ПАРАМЕТРЫ УЧИТЕЛЯ (ПИД) ==========
# Интегральная часть учителя:
#   'true'   - настоящий интегратор: I += Ki * e на каждом такте (антивиндап, |I| <= I_MAX); при тренде (S1)
#              статической ошибки нет. Ki - прирост за такт, PID_WINDOW не используется;
#   'window' - прежняя: Ki * (сумма e за последние PID_WINDOW тактов). Это по сути ещё один пропорциональный
#              коэффициент Kp + Ki * PID_WINDOW, поэтому при тренде остаётся статическая ошибка.
# Переключить без правки файла: переменная окружения NIR_PID_INT=window.
PID_INTEGRATOR = os.environ.get('NIR_PID_INT', 'true').strip().lower()   # читаем переменную, по умолчанию 'true'
assert PID_INTEGRATOR in ('true', 'window'), "NIR_PID_INT должна быть 'true' или 'window'"   # защита от опечатки

# Ручные значения (KP повышен с 1.2 до 1.6). Если запущен tune_pid.py, значения берутся из tuned/pid_int.json
# (интегратор) или tuned/pid.json (окно): коэффициенты Ki у этих режимов разного смысла, файлы не смешиваются.
KP = 1.6
KD = 0.20
KI = 0.10 if PID_INTEGRATOR == 'true' else 0.05   # для интегратора Ki - прирост за такт, поэтому другое значение
I_MAX = 0.4
# Разделение интеграла (режим 'true'): интегратор копит ошибку, только пока |e| <= I_SEP. Иначе после скачка он
# набирает лишнее на большой ошибке и даёт перерегулирование. Больше I_SEP - нулевая ошибка и на более крутых
# трендах, но больше перерегулирование на скачках; при I_SEP = 0.2 нулевая ошибка держится до тренда ~1,6 x S1.
I_SEP = 0.2
_pid = _load('pid_int.json' if PID_INTEGRATOR == 'true' else 'pid.json')   # свой файл настроек для каждого режима
KP = _pid.get('KP', KP)
KI = _pid.get('KI', KI)
KD = _pid.get('KD', KD)
PID_WINDOW = int(_pid.get('PID_WINDOW', PID_WINDOW))

# ========== ФУНКЦИЯ ПОТЕРЬ ИМИТАЦИИ ПИД-ЭТАЛОНА (предобучение RL-регулятора, rl_core.PPO.pretrain_imitation) ==========
# Параметры самого RL (PPO, среды, награда, архитектура) - в config_rl.py.
ACTIVE_W = 5.0
QUIET_BAND = 0.06     # цель учителя |u| <= QUIET_BAND - канал в норме
FA_MARGIN = 0.08      # на таких каналах штраф за |u| > FA_MARGIN
FA_W = 20.0

# ========== ПОРОГИ ==========
IDLE_THRESHOLD = 0.10

# ========== СЦЕНАРИИ ==========
SCENARIOS = {
    'S4': {'description': 'Без особой причины (контрольный сценарий)', 'rate_param': None, 'rate_value': 0.0, 'jump_param': None, 'jump_value': 0.0},
    'S1': {'description': 'Тренд x1 (износ инструмента, токарная обработка)', 'rate_param': 0, 'rate_value': 0.027, 'jump_param': None, 'jump_value': 0.0},
    'S2': {'description': 'Скачок x6 (вибрация, фрезерная обработка)', 'rate_param': None, 'rate_value': 0.0, 'jump_param': 5, 'jump_value': 1.6},
    'S3': {'description': 'Скачок x8 (недостаточное охлаждение, закалка)', 'rate_param': None, 'rate_value': 0.0, 'jump_param': 7, 'jump_value': -1.6}
}

CRITICAL_SCENARIOS = {
    'C1': {'description': 'Мощный скачок x1', 't0': 60, 'noise_scale': 1.0, 'components': [{'kind': 'jump', 'param': 0, 'value': 5.0}]},
    'C2': {'description': 'Крутой тренд x6', 't0': 65, 'noise_scale': 1.0, 'components': [{'kind': 'trend', 'param': 5, 'value': 0.12}]},
    'C3': {'description': 'Скачок на трёх операциях', 't0': 70, 'noise_scale': 1.0, 'components': [{'kind': 'jump', 'param': 1, 'value': 4.0}, {'kind': 'jump', 'param': 4, 'value': 4.0}, {'kind': 'jump', 'param': 6, 'value': 4.0}]},
    'C4': {'description': 'Неуправляемый дрейф', 't0': 60, 'noise_scale': 1.0, 'components': [{'kind': 'trend', 'param': 0, 'value': 0.15}, {'kind': 'trend', 'param': 2, 'value': 0.15}, {'kind': 'trend', 'param': 3, 'value': 0.15}, {'kind': 'trend', 'param': 6, 'value': 0.15}]},
    'C5': {'description': 'Авария: скачок всех показателей', 't0': 75, 'noise_scale': 1.0, 'components': [{'kind': 'jump', 'param': i, 'value': 4.0} for i in range(N_PARAMS)]},
    'C6': {'description': 'Скачок и тренд (x3)', 't0': 65, 'noise_scale': 1.0, 'components': [{'kind': 'jump', 'param': 2, 'value': 3.0}, {'kind': 'trend', 'param': 2, 'value': 0.08}]},
    'C7': {'description': 'Импульсный выброс x5 и x8', 't0': 80, 'noise_scale': 1.0, 'components': [{'kind': 'pulse', 'param': 4, 'value': 6.0, 'duration': 8}, {'kind': 'pulse', 'param': 7, 'value': 6.0, 'duration': 8}]},
    'C8': {'description': 'Вибрация: колебания x4 и x6', 't0': 60, 'noise_scale': 1.0, 'components': [{'kind': 'sine', 'param': 3, 'value': 2.0, 'period': 12}, {'kind': 'sine', 'param': 5, 'value': 2.0, 'period': 12}]},
    'C9': {'description': 'Каскад скачков', 't0': 65, 'noise_scale': 10.0, 'components': [{'kind': 'jump', 'param': 0, 'value': 3.0, 'start': 0}, {'kind': 'jump', 'param': 4, 'value': -3.0, 'start': 12}, {'kind': 'jump', 'param': 7, 'value': 3.0, 'start': 24}]},
}

SEED = 42
SEED_TEACHER_START = 1000
SEED_EXPERIMENT_START = 7000
SEED_CRITICAL_START = 50000
SEED_FALSE_ALARM_START = 90000