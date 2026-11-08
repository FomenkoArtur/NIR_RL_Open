NOMENCLATURE = [
    {"key": "shaft", "name": "Вал", "description": "Токарная обработка"},
    {"key": "flange", "name": "Фланец", "description": "Фрезерная обработка"},
    {"key": "housing", "name": "Корпус", "description": "Закалка"},
]

# Разные физические параметры (a - инерция, c - эффективность привода)
PLANT_PARAMS = {
    "shaft":   {"a": 0.99, "c": 0.10},  # Номинал
    "flange":  {"a": 0.98, "c": 0.15},  # Легкий, быстрый
    "housing": {"a": 0.995, "c": 0.05}, # Тяжелый, инертный
}

# Три абсолютно разных PID
PID_GAINS = {
    "shaft":   {"kp": 1.5, "ki": 0.2, "kd": 0.1},
    "flange":  {"kp": 3.0, "ki": 0.5, "kd": 0.2},
    "housing": {"kp": 5.0, "ki": 1.0, "kd": 0.5},
}

# Для afc_bode.py (графики АФХ/Боде)
NOMINAL_PARAMS = {
    "shaft":   {"k": 1.00, "tau": 1.0},
    "flange":  {"k": 1.50, "tau": 0.5},
    "housing": {"k": 0.50, "tau": 2.0},
}

def items():
    return NOMENCLATURE

def names():
    return [item["name"] for item in NOMENCLATURE]

def get_pid_gains(key):
    return PID_GAINS[key]

def nominal_params(key):
    return dict(NOMINAL_PARAMS.get(key, NOMINAL_PARAMS["shaft"]))