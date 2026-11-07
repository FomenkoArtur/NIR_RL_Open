import config as C
from config import ZONES
from config_rl import RL, RL_ARCH, RL_TRAIN

PARAMETERS = [
    ("x₁", "Отклонение диаметра от номинала", "Токарная обработка"),
    ("x₂", "Шероховатость Ra", "Токарная обработка"),
    ("x₃", "Вибрация в зоне резания", "Токарная обработка"),
    ("x₄", "Отклонение размера от номинала", "Фрезерная обработка"),
    ("x₅", "Шероховатость Ra", "Фрезерная обработка"),
    ("x₆", "Вибрация в зоне резания", "Фрезерная обработка"),
    ("x₇", "Температура нагрева", "Закалка"),
    ("x₈", "Скорость охлаждения", "Закалка"),
]

CONFIG_ITEMS = [                                   # config.py
    ("G", "Программное задание g"),
    ("DELTA", "Допустимая полоса δ"),
    ("N_WINDOW", "Окно нормализации L_N, тактов"),
    ("GAMMA", "Коэффициент забывания γ"),
    ("S_MIN_FRAC", "Нижняя граница масштаба s_min, доля H_i"),
    ("BETA", "Сдвиг β критериев (формула (3))"),
    ("WINDOW", "Длина окна наблюдения сети L, тактов"),
    ("T_START", "Такт возникновения особой причины"),
    ("T_OBS", "Горизонт наблюдения, тактов"),
    ("N_CONS", "Тактов подряд в полосе для Ts"),
    ("N_REAL", "Реализаций на сценарий"),
    ("IDLE_THRESHOLD", "Порог холостого хода |u|"),
    ("KP", "ПИД-эталон: Kp"),
    ("KI", "ПИД-эталон: Ki"),
    ("KD", "ПИД-эталон: Kd"),
    ("SEED", "Начальное зерно ГСЧ"),
    ("CONTROL_LAYOUT", "Схема адресации воздействия"),
    ("N_ZONES", "Число каналов управления"),
]

RL_ITEMS = [                                       # config_rl.py: (имя, функция значения, описание)
    ("RL.ALGO", lambda: RL['ALGO'], "Алгоритм обучения с подкреплением"),
    ("RL.FEAT", lambda: RL['FEAT'], "Вход сети: u = (ε, Nᵢ, uᵢ), legacy = (ε, Nᵢ, wᵢNᵢ)"),
    ("RL.REF.MODE", lambda: RL['REF']['MODE'], "Вид эталонной динамики c_ref"),
    ("RL.IMIT.ENABLED", lambda: RL['IMIT']['ENABLED'], "Предобучение имитацией ПИД-эталона"),
    ("RL.PPO.ITERS", lambda: RL['PPO']['ITERS'], "Итераций PPO при полном обучении"),
    ("RL.PPO.N_ENVS", lambda: RL['PPO']['N_ENVS'], "Параллельных сред по типам устройств"),
    ("RL.PPO.GAMMA", lambda: RL['PPO']['GAMMA'], "Коэффициент дисконтирования"),
    ("RL.PPO.LAMBDA", lambda: RL['PPO']['LAMBDA'], "Параметр GAE"),
    ("RL.PPO.CLIP", lambda: RL['PPO']['CLIP'], "Отсечение отношения вероятностей"),
    ("RL.PPO.EPOCHS", lambda: RL['PPO']['EPOCHS'], "Проходов по данным за итерацию"),
    ("RL_TRAIN", lambda: RL_TRAIN, "LR, WEIGHT_DECAY, BATCH_SIZE (tuned/cnn_rl.json)"),
    ("RL_ARCH", lambda: RL_ARCH, "Архитектура сети регулятора"),
]


def config_rows():
    rows = [(name, getattr(C, name), description) for name, description in CONFIG_ITEMS]
    return rows + [(name, getter(), description) for name, getter, description in RL_ITEMS]


def zone_labels():
    labels = []
    for indices in ZONES:
        names = [PARAMETERS[i][0] for i in indices]
        labels.append(f"u({names[0]})" if len(names) == 1 else f"u({names[0]}…{names[-1]})")
    return labels