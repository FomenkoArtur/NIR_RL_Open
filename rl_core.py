"""
rl_core.py - RL-контур: двойник с рандомизацией (пакет сред на устройстве), эталонная модель, награда,
сеть регулятора с настраиваемой архитектурой, ПИД на устройстве, PPO, метрики замкнутого контура и отчёт.
Тело файла между маркерами BEGIN/END вставляется в tune_cnn.py без изменений; здесь сверху только импорты.
"""

import math                                                     # log(2*pi) для нормального распределения
import time                                                     # замер времени вывода сети

import numpy as np                                              # генерация эпизодов (воспроизводимо на любом устройстве)
import torch                                                    # расчёт сред и сетей на устройстве
import torch.nn as nn                                           # слои сети
import torch.optim as optim                                     # Adam

from config import (A_CUR, C_RAW_FACTOR, CRITICAL_SCENARIOS, DELTA, G, IDLE_THRESHOLD, IQR_TO_SIGMA, N_CONS,
                    N_FEATURES, N_PARAMS, N_WINDOW, SCENARIOS, SEED_CRITICAL_START, SIG_CUR, T_OBS, T_START,
                    T_TOTAL, WINDOW)                            # те же константы, что у прежнего кода
from config_rl import RL, RL_ARCH                               # конфигурация RL и фиксированная архитектура
from fastsim import episode_noise                               # тот же шум эпизода по зерну, что раньше
from plant import plant                                         # калибровка карт и веса (Plant(seed=42))
from scenarios import critical_profile, custom_profile          # профили C1-C9 и тренд/скачок на показатель
from trainer import weighted_mse                                # функция потерь имитации - та же, что сейчас

X_BAR, S_SCALE = plant.x_bar, plant.S                           # центр карты и ширина коридора H_i
LCL, UCL, S_MIN = plant.LCL, plant.UCL, plant.S_MIN             # границы карты и нижняя граница масштаба s_min
TREND, JUMP = SCENARIOS['S1']['rate_value'], abs(SCENARIOS['S2']['jump_value'])   # амплитуды S1 и S2 (как в tune_cnn)


# >>> BEGIN RL-CORE (вставляется в tune_cnn.py без изменений)
D_MAX = 8                                                       # запас буфера команд под задержку (OOD - до 4 тактов)
N_STEPS = T_TOTAL - 1                                           # решений за эпизод: на последнем такте u уже не влияет
PRIV_DIM = 16                                                   # длина вектора параметров объекта для критика
LOG2PI = math.log(2 * math.pi)                                  # константа в логарифме плотности нормального закона


def _dt(dev):
    """Тип чисел объекта: double, как у прежнего Sim; на XPU (нет быстрого fp64) - float."""
    return torch.float32 if dev.type == 'xpu' else torch.float64   # объект и нормализация считаются в double


def ref_dim():
    """Длина вектора c_ref: (T_ref, sigma_ref) или (Kp, Ki, Kd)."""
    return 3 if RL['REF']['MODE'] == 'pid' else 2               # зависит от выбранного вида эталона


def extra_dim(head):
    """Длина вектора, подаваемого в FC мимо свёрток: c_ref (+ u_ПИД для режима 'ПИД + поправка')."""
    return ref_dim() + (N_PARAMS if head == 'pid_res' else 0)   # u_ПИД нужен сети, чтобы знать, что она поправляет


# ---------------------------- генерация эпизодов (numpy, на CPU, воспроизводимо) ----------------------------
def _val(rng, v, size=None):
    """Скаляр - как есть; пара (lo, hi) - равномерно из диапазона."""
    if isinstance(v, (tuple, list)):                            # задан диапазон рандомизации
        return rng.uniform(v[0], v[1], size)                    # равномерно, как в постановке
    return np.full(size, float(v)) if size else float(v)        # фиксированное значение (номинал, OOD)


def _plant(rng, spec):
    """Параметры одного объекта по спецификации (скаляры или диапазоны)."""
    d = spec['DELAY']                                           # задержка: целое или диапазон целых
    d = int(rng.randint(d[0], d[1] + 1)) if isinstance(d, (tuple, list)) else int(d)   # 1..3 включительно
    cross = float(spec['CROSS'])                                # амплитуда перекрёстных связей привода
    off = np.zeros((N_PARAMS, N_PARAMS))                        # внедиагональная часть матрицы B
    if cross > 0:                                               # ГСЧ тратится только при наличии связей
        off = rng.uniform(-cross, cross, (N_PARAMS, N_PARAMS)) * (1.0 - np.eye(N_PARAMS))   # диагональ не трогаем
    return {
        'a': _val(rng, spec['A']),                              # инерция объекта a
        'c': _val(rng, spec['C_FRAC'], N_PARAMS) * S_SCALE,     # эффективность привода c_i = доля * H_i (своя у каждого)
        'd': d,                                                 # запаздывание, тактов
        'ns': _val(rng, spec['NOISE']),                         # множитель sigma шума
        'B': np.eye(N_PARAMS) + off,                            # матрица адресации: единичная + малые связи
        'xm': float(np.abs(off).sum() / (N_PARAMS * (N_PARAMS - 1))),   # средний модуль связи (признак для критика)
        'dz': _val(rng, spec['DEAD']),                          # мёртвая зона привода по |u|
        'slo': _val(rng, spec['SAT_LO']),                       # насыщение снизу: u >= -slo
        'shi': _val(rng, spec['SAT_HI']),                       # насыщение сверху: u <= shi
    }


def _disturbance(rng, a, dist):
    """Одно возмущение: (rate (T, 8), jump (8,), t0, quiet) в абсолютных единицах (умножено на H_i)."""
    rate, jump = np.zeros((T_TOTAL, N_PARAMS)), np.zeros(N_PARAMS)   # добавка на такт и скачок на такте t0
    kinds = list(dist['P_KIND'])                                # виды возмущений
    p = np.array([dist['P_KIND'][k] for k in kinds], dtype=float)   # их вероятности
    kind = kinds[rng.choice(len(kinds), p=p / p.sum())]         # выбор вида эпизода
    t0 = int(rng.randint(dist['T0'][0], dist['T0'][1] + 1))     # такт начала возмущения
    par = int(rng.randint(N_PARAMS))                            # показатель, на который оно действует
    sgn = float(rng.choice([-1.0, 1.0]))                        # знак возмущения
    if kind == 'trend':                                         # линейный тренд (износ инструмента, скорость дрейфа r)
        rate[t0:, par] = sgn * rng.uniform(*dist['TREND'])      # постоянная добавка на каждом такте после t0
    elif kind == 'jump':                                        # скачок
        jump[par] = sgn * rng.uniform(*dist['JUMP'])            # однократный сдвиг состояния на t0
    elif kind == 'pulse':                                       # импульс: выброс и возврат через dur тактов
        dur = int(rng.randint(dist['PULSE_DUR'][0], dist['PULSE_DUR'][1] + 1))   # длительность импульса
        v = sgn * rng.uniform(*dist['PULSE'])                   # амплитуда импульса
        rate[t0, par] += v                                      # выброс на t0
        if t0 + dur < T_TOTAL:                                  # возврат помещается в эпизод
            rate[t0 + dur, par] -= v * a ** dur                 # с учётом инерции именно этого объекта (как C7)
    elif kind == 'sine':                                        # колебание (вибрация)
        per = int(rng.randint(dist['SINE_PERIOD'][0], dist['SINE_PERIOD'][1] + 1))   # период, тактов
        v = sgn * rng.uniform(*dist['SINE'])                    # амплитуда колебания состояния
        k = np.arange(T_TOTAL - t0)                             # такты от начала колебания
        rate[t0:, par] += v * (2 * np.pi / per) * np.cos(2 * np.pi * k / per)   # производная синуса, как в critical_profile
    elif kind == 'multi':                                       # одновременные скачки нескольких показателей
        n = int(rng.randint(dist['MULTI_N'][0], dist['MULTI_N'][1] + 1))   # сколько показателей
        idx = rng.choice(N_PARAMS, n, replace=False)            # какие именно (без повторов)
        jump[idx] = rng.choice([-1.0, 1.0], n) * rng.uniform(*dist['JUMP'], n) * dist['MULTI_SCALE']   # свои знаки и амплитуды
    return rate * S_SCALE, jump * S_SCALE, t0, kind == 'none'   # в абсолютные единицы, как custom_profile


def _cref(rng):
    """Случайный вектор c_ref эпизода (сырые значения)."""
    r = RL['REF']                                               # настройки эталона
    if r['MODE'] == 'pid':                                      # эталон - ПИД: множители к Kp, Ki, Kd
        return rng.uniform(r['PID_SCALE'][0], r['PID_SCALE'][1], 3)
    return np.array([rng.uniform(*r['TAU']), rng.uniform(*r['OS'])])   # эталон - (T_ref, допустимое перерегулирование)


def _cref_eval():
    """Фиксированный c_ref для оценки: все кандидаты и baseline сравниваются при одном и том же задании."""
    r = RL['REF']                                               # настройки эталона
    return np.array(r['EVAL_PID'] if r['MODE'] == 'pid' else r['EVAL_TAU'], dtype=float)


def cref_features(cref):
    """Сырые c_ref -> признаки в [-1; 1] для сети (тензор (B, R))."""
    r = RL['REF']                                               # диапазоны эталона
    if r['MODE'] == 'pid':                                      # множители 0,5..1,5 -> -1..1
        lo, hi = r['PID_SCALE']
        return (cref - (lo + hi) / 2) / ((hi - lo) / 2)
    lo = torch.tensor([r['TAU'][0], r['OS'][0]], dtype=cref.dtype, device=cref.device)   # нижние границы диапазонов
    hi = torch.tensor([r['TAU'][1], r['OS'][1]], dtype=cref.dtype, device=cref.device)   # верхние границы
    return (cref - (lo + hi) / 2) / ((hi - lo) / 2)             # линейное отображение в [-1; 1]


def _pack(items):
    """Список эпизодов (словари) -> словарь массивов с первой осью = номер эпизода."""
    out = {k: np.stack([np.asarray(it[k]) for it in items]) for k in items[0] if k != 'name'}   # числовые поля
    out['names'] = [it.get('name', '') for it in items]         # подписи эпизодов для разбивки отчёта
    return out


def sample_episodes(rng, n, spec=None):
    """n случайных эпизодов: объект из spec (по умолчанию диапазоны DR), возмущение, шум, c_ref."""
    spec = spec or RL['DR']                                     # по умолчанию - обучающие диапазоны
    items = []                                                  # эпизоды по одному
    for _ in range(n):
        pl = _plant(rng, spec)                                  # параметры объекта
        rate, jump, t0, quiet = _disturbance(rng, pl['a'], RL['DIST'])   # возмущение
        noise = rng.normal(0.0, SIG_CUR, (T_TOTAL - 1, N_PARAMS)) * pl['ns']   # шум с масштабом этого объекта
        items.append(dict(pl, rate=rate, jump=jump, t0=t0, quiet=quiet, noise=noise, cref=_cref(rng), name='dr'))
    return _pack(items)                                         # одним словарём массивов


def suite_nominal():
    """Номинальный объект: тренд и скачок на каждый показатель (как build_cases) + эпизоды без возмущений."""
    seed0 = RL['SEEDS']['NOMINAL']                              # свои зёрна шума (не пересекаются с прежними)
    nom = _plant(np.random.RandomState(seed0), RL['NOMINAL'])   # номинал: случайных элементов в нём нет
    items, k = [], 0                                            # эпизоды и номер зерна
    for p in range(N_PARAMS):                                   # каждый из 8 показателей
        for kind, val in (('trend', TREND), ('jump', JUMP)):    # те же случаи, что в tune_pid / tune_cnn
            sign = 1 if (p + (kind == 'jump')) % 2 == 0 else -1   # знак чередуется
            rate, jump = custom_profile(p, kind, sign * val, T_START)   # профиль на номинальном t0
            items.append(dict(nom, rate=rate, jump=jump, t0=T_START, quiet=False, noise=episode_noise(seed0 + k),
                              cref=_cref_eval(), name=f'{kind} x{p + 1}'))
            k += 1
    for i in range(RL['SEARCH']['N_QUIET']):                    # без возмущений: проверка ложных срабатываний
        items.append(dict(nom, rate=np.zeros((T_TOTAL, N_PARAMS)), jump=np.zeros(N_PARAMS), t0=T_START, quiet=True,
                          noise=episode_noise(seed0 + 5000 + i), cref=_cref_eval(), name='S4'))
    return _pack(items)


def suite_dr(n=None, seed=None):
    """Фиксированная выборка рандомизированных объектов (одна и та же для всех кандидатов)."""
    n = n or RL['SEARCH']['N_EVAL_DR']                          # объём выборки
    ep = sample_episodes(np.random.RandomState(seed or RL['SEEDS']['EVAL']), n)   # своё зерно - выборка неизменна
    ep['cref'][:] = _cref_eval()                                # c_ref при оценке фиксирован
    return ep


def suite_ood():
    """Объекты вне обучающих диапазонов: каждый из RL['OOD'] x N_OOD_EP случайных возмущений."""
    rng = np.random.RandomState(RL['SEEDS']['OOD'])             # своё зерно
    items = []                                                  # эпизоды
    for spec in RL['OOD']:                                      # каждый «чужой» объект
        full = dict(RL['NOMINAL'], **{k: v for k, v in spec.items() if k != 'name'})   # номинал + отклонение
        pl = _plant(rng, full)                                  # его параметры
        for _ in range(RL['SEARCH']['N_OOD_EP']):               # несколько возмущений на объект
            quiet = True                                        # на OOD-объектах нужны именно переходные процессы
            while quiet:                                        # вид 'none' перевыбирается
                rate, jump, t0, quiet = _disturbance(rng, pl['a'], RL['DIST'])
            noise = rng.normal(0.0, SIG_CUR, (T_TOTAL - 1, N_PARAMS)) * pl['ns']   # шум объекта
            items.append(dict(pl, rate=rate, jump=jump, t0=t0, quiet=False, noise=noise, cref=_cref_eval(),
                              name=spec['name']))
    return _pack(items)


def suite_critical():
    """C1-C9 на номинальном объекте - только отложенная проверка, в обучении не участвуют."""
    nom = _plant(np.random.RandomState(0), RL['NOMINAL'])       # номинальный объект
    items = []                                                  # девять эпизодов
    for ep, (name, spec) in enumerate(CRITICAL_SCENARIOS.items(), start=1):
        rate, jump = critical_profile(name)                     # профиль ситуации из config
        noise = episode_noise(SEED_CRITICAL_START + ep, spec.get('noise_scale', 1.0))   # тот же шум, что в тестовой выборке
        items.append(dict(nom, rate=rate, jump=jump, t0=spec['t0'], quiet=False, noise=noise,
                          cref=_cref_eval(), name=name))
    return _pack(items)


def to_device(ep, dev):
    """Перенос эпизодов на устройство одним разом (дальше пересылок по тактам нет)."""
    dt = _dt(dev)                                               # точность объекта
    f = lambda k: torch.as_tensor(ep[k], dtype=dt, device=dev)  # вещественное поле -> тензор
    out = {k: f(k) for k in ('c', 'B', 'rate', 'jump', 'noise', 'cref')}   # массивы
    for k in ('a', 'ns', 'dz', 'slo', 'shi', 'xm'):             # скаляры эпизода -> столбец (B, 1) для broadcasting
        out[k] = f(k).view(-1, 1)
    out['d'] = torch.as_tensor(ep['d'], dtype=torch.long, device=dev)    # задержка (индекс в истории команд)
    out['t0'] = torch.as_tensor(ep['t0'], dtype=torch.long, device=dev)  # такт возмущения
    out['np'] = {'t0': np.asarray(ep['t0']), 'quiet': np.asarray(ep['quiet']), 'names': ep['names']}   # для метрик на CPU
    return out


# ---------------------------- ПИД на устройстве (как BatchPIDPolicy) ----------------------------
class PIDTorch:
    """ПИД по каждой зоне (зона = показатель, V = I) для B сред; коэффициенты - числа или столбцы (B, 1)."""

    def __init__(self, batch, dev, dt, kp, ki, kd):
        cfg = RL['PID']                                         # режим и ограничения - из конфигурации
        self.mode, self.imax, self.isep = cfg['mode'], cfg['I_MAX'], cfg['I_SEP']   # интегратор, |I| <= I_MAX, I_SEP
        self.kp, self.ki, self.kd = kp, ki, kd                  # коэффициенты (общие или свои у каждой среды)
        self.n_i = int(cfg['WINDOW'])                           # окно суммы (режим 'window')
        self.I = torch.zeros((batch, N_PARAMS), dtype=dt, device=dev)       # состояние настоящего интегратора
        self.e_prev = torch.zeros((batch, N_PARAMS), dtype=dt, device=dev)  # ошибка прошлого такта (D-часть)
        self.hist = torch.zeros((batch, self.n_i, N_PARAMS), dtype=dt, device=dev)   # кольцевое окно ошибок
        self.k = 0                                              # номер такта для кольцевого окна

    def act(self, x_norm):
        e = x_norm - G                                          # ошибка зоны = N_i - g (V = I при CONTROL_LAYOUT='parameter')
        zero = torch.zeros_like(e)                              # нули нужной формы
        if self.mode == 'true':                                 # настоящий интегратор (config.PID_INTEGRATOR = 'true')
            p = torch.where(e.abs() <= DELTA, zero, self.kp * e + self.kd * (e - self.e_prev))   # мёртвая зона на P и D
            u_raw = p + self.I                                  # интегральная часть добавляется всегда
            u = torch.clamp(u_raw, -1.0, 1.0)                   # диапазон привода
            sat = ((u_raw > 1.0) & (e > 0)) | ((u_raw < -1.0) & (e < 0))   # привод упёрся, ошибка того же знака
            freeze = sat | (e.abs() > self.isep)                # антивиндап и разделение интеграла
            self.I = torch.clamp(self.I + torch.where(freeze, zero, self.ki * e), -self.imax, self.imax)
            self.e_prev = e                                     # для D-части следующего такта
            return u
        pos = self.k % self.n_i                                 # режим 'window': сумма за последние n_i тактов
        self.hist[:, pos] = e                                   # запись ошибки в кольцо
        i_term = torch.clamp(self.ki * self.hist.sum(dim=1), -self.imax, self.imax)   # «интеграл» по окну
        u_raw = self.kp * e + self.kd * (e - self.e_prev) + i_term   # ПИД-закон
        u = torch.clamp(u_raw, -1.0, 1.0)                       # диапазон привода
        sat = ((u_raw > 1.0) & (e > 0)) | ((u_raw < -1.0) & (e < 0))   # насыщение
        self.hist[:, pos] = torch.where(sat, zero, e)           # антиприращение интеграла
        self.e_prev, self.k = e, self.k + 1                     # к следующему такту
        return torch.where(e.abs() <= DELTA, zero, u)           # мёртвая зона


def pid_for(ep, dev, dt):
    """ПИД эпизода: номинальные коэффициенты, а в режиме эталона 'pid' - умноженные на c_ref."""
    cfg, batch = RL['PID'], ep['a'].shape[0]                    # коэффициенты и число сред
    kp, ki, kd = cfg['KP'], cfg['KI'], cfg['KD']                # номинальные коэффициенты
    if RL['REF']['MODE'] == 'pid':                              # эталон задан коэффициентами ПИД
        sc = ep['cref']                                         # (B, 3) множители эпизода
        kp, ki, kd = kp * sc[:, 0:1], ki * sc[:, 1:2], kd * sc[:, 2:3]   # свои коэффициенты у каждой среды
    return PIDTorch(batch, dev, dt, kp, ki, kd)


# ---------------------------- пакет сред с рандомизацией ----------------------------
class RLEnv:
    """
    B сред одновременно на устройстве. Порядок такта как в Sim.run: скачок на t0, объект (34) с задержкой,
    мёртвой зоной, насыщением и перекрёстными связями привода, нормализация (10)-(12).
    """

    def __init__(self, dev, reward=True, pid_inner=False, shadow=False):
        self.dev, self.dt = dev, _dt(dev)                       # устройство и точность объекта
        self.reward_on, self.pid_inner, self.is_shadow = reward, pid_inner, shadow   # награда / внутренний ПИД / «тень»
        as_t = lambda v: torch.as_tensor(np.asarray(v), dtype=self.dt, device=dev)   # константа -> тензор
        self.x_bar, self.lcl, self.ucl = as_t(X_BAR), as_t(LCL), as_t(UCL)   # калибровка карт
        self.s_min, self.S = as_t(S_MIN), as_t(S_SCALE)         # s_min и ширина коридора H_i
        self.offsets = torch.arange(WINDOW, device=dev) - (WINDOW - 1)        # смещения окна входа от текущего такта

    def reset(self, ep):
        dev, dt = self.dev, self.dt                             # короткие имена
        self.ep, self.B, self.t = ep, ep['a'].shape[0], 0       # эпизоды, число сред, такт
        b = self.B
        self.buf = torch.full((b, N_PARAMS, N_WINDOW), float('nan'), dtype=dt, device=dev)   # окно L_N нормализации
        self.valid = torch.zeros((b, N_PARAMS), dtype=torch.long, device=dev)  # принятых значений в окне
        self.m = self.s = None                                  # оценки медианы и масштаба
        self.state = self.x_bar.repeat(b, 1)                    # x(0) = центр карты
        self.H = torch.zeros((b, T_TOTAL + D_MAX, N_PARAMS), dtype=dt, device=dev)   # история команд со сдвигом D_MAX
        self.E = torch.empty((b, T_TOTAL), dtype=dt, device=dev)                     # среднее отклонение от задания
        self.N = torch.empty((b, T_TOTAL, N_PARAMS), dtype=dt, device=dev)           # N_i(t)
        self.u_last = torch.zeros((b, N_PARAMS), dtype=dt, device=dev)               # последняя команда
        self.held = torch.zeros(b, dtype=torch.long, device=dev)                     # тактов подряд с |eps| <= delta
        self.e_peak = torch.zeros(b, dtype=dt, device=dev)                           # пик текущего отклонения
        self.cref_f = cref_features(ep['cref']).float()         # признаки c_ref для сети
        if RL['REF']['MODE'] == 'tau':                          # параметры эталона 'tau'
            self.rho = torch.exp(-1.0 / ep['cref'][:, 0])       # желаемое сжатие ошибки за такт exp(-1/T_ref)
            self.os_ref = ep['cref'][:, 1]                      # допустимое перерегулирование (доля пика)
        self.pid = pid_for(ep, dev, dt) if self.pid_inner else None   # внутренний ПИД: учитель и 'ПИД + поправка'
        self.u_pid = None                                       # его выход на текущем такте
        self.shadow = None                                      # «тень»: тот же объект под эталонным ПИД (режим 'pid')
        if self.reward_on and RL['REF']['MODE'] == 'pid' and not self.is_shadow:
            self.shadow = RLEnv(dev, reward=False, pid_inner=True, shadow=True)   # своя копия среды без награды
            self.shadow.reset(ep)                               # тот же объект, возмущение и шум
        self._jump()                                            # скачок на t0 = 0 (если вдруг задан)
        self._observe()                                         # N(0), eps(0)

    def _jump(self):
        hit = (self.ep['t0'] == self.t)[:, None]                # среды, где возмущение начинается сейчас
        self.state = self.state + torch.where(hit, self.ep['jump'], torch.zeros_like(self.ep['jump']))   # скачок состояния

    def _observe(self):
        """Нормализация (10)-(12) - тот же код, что в Sim.run (tune_cnn.py); ошибка - среднее отклонение N_i от G."""
        st, t = self.state, self.t                              # состояние и такт
        admitted = ((st >= self.lcl) & (st <= self.ucl)) | (self.valid == 0)   # в окно - только внутри карты (или окно пусто)
        row = torch.where(admitted, st, torch.full_like(st, float('nan')))     # не принятые - NaN
        pos = t % N_WINDOW                                      # позиция в кольцевом окне
        self.valid = self.valid - (~torch.isnan(self.buf[:, :, pos])).long()   # уходящее значение
        self.buf[:, :, pos] = row                               # запись нового
        self.valid = self.valid + (~torch.isnan(row)).long()    # пришедшее значение
        srt = torch.sort(self.buf, dim=2).values                # NaN уходят в конец
        q = lambda r: torch.gather(srt, 2, torch.clamp(r - 1, min=0).unsqueeze(2)).squeeze(2)   # порядковая статистика номера r
        v = self.valid
        q25, q50, q75 = q((v + 3) // 4), q((v + 1) // 2), q((3 * v + 3) // 4)   # квартили
        m_new, s_new = q50, torch.maximum((q75 - q25) / IQR_TO_SIGMA, self.s_min)   # формула (11)
        if self.m is None:                                      # первый такт
            self.m, self.s = m_new, s_new
        else:
            empty = v == 0                                      # окно пусто - оценки не меняются
            self.m, self.s = torch.where(empty, self.m, m_new), torch.where(empty, self.s, s_new)
        x_norm = 0.5 * (1.0 + torch.tanh(0.5 * (st - self.m) / self.s))   # N_i(t), формула (12)
        self.N[:, t] = x_norm                                   # 8 нормализованных показателей - вход сети
        self.E[:, t] = G - x_norm.mean(dim=1)                   # средняя ошибка по показателям, без весов w_i
        if self.pid is not None:                                # внутренний ПИД вызывается ровно раз за такт
            self.u_pid = self.pid.act(x_norm)

    def window(self, feat=None):
        """Вход сети: (B, 8, WINDOW), float32 - 8 нормализованных показателей N_i; окно слева дополняется тактом 0."""
        idx = torch.clamp(self.offsets + self.t, min=0)         # такты окна
        n = self.N[:, idx].float()                              # N_i в окне, (B, WINDOW, 8)
        return ((n - 0.5) * 2.0).transpose(1, 2)                # (B, 8, WINDOW), те же масштабы, что в CNNPolicy

    def extra(self, head):
        """Вектор мимо свёрток: c_ref (+ u_ПИД)."""
        if head == 'pid_res':                                   # сеть поправляет ПИД - должна видеть его выход
            return torch.cat([self.cref_f, self.u_pid.float()], dim=1)
        return self.cref_f                                      # иначе только c_ref

    def priv(self):
        """Параметры объекта для критика (в обучении известны, у актора их нет)."""
        ep = self.ep
        p = torch.cat([(ep['a'] - 0.97) / 0.03,                 # инерция (приведена к ~[-1; 1])
                       (ep['c'] / self.S - 0.125) / 0.075,      # c_i / H_i (8 значений)
                       (ep['d'].view(-1, 1).to(self.dt) - 2.0),  # задержка
                       torch.log(ep['ns']),                     # шум (логарифм множителя)
                       ep['xm'] * 20.0,                         # перекрёстные связи
                       ep['dz'] * 20.0,                         # мёртвая зона
                       ep['slo'], ep['shi'],                    # насыщение
                       torch.full_like(ep['a'], self.t / T_TOTAL)], dim=1)   # время: эпизод ограничен по длине
        return p.float()

    def step(self, u):
        """Применить команду u(t), перейти к t+1; вернуть награду (B,) или None."""
        u = u.to(self.dt)                                       # команда в точности объекта
        e_old = self.E[:, self.t]                               # eps до шага (для эталона 'tau')
        self.H[:, self.t + D_MAX] = u                           # в историю команд
        du = u - self.u_last                                    # приращение команды (штраф за рывки)
        self.u_last = u                                         # запомнить для следующего такта
        if self.shadow is not None:                             # эталонный ПИД делает свой шаг на «тени»
            self.shadow.step(self.shadow.u_pid)
        self.t += 1                                             # следующий такт
        t, ep = self.t, self.ep
        self._jump()                                            # скачок на t0 - до шага объекта, как в Sim.run
        idx = (t - ep['d'] + D_MAX).view(-1, 1, 1).expand(-1, 1, N_PARAMS)   # команда, поданная d тактов назад
        u_app = self.H.gather(1, idx).squeeze(1)                # дошедшая до объекта команда
        u_act = torch.sign(u_app) * torch.relu(u_app.abs() - ep['dz'])        # мёртвая зона привода
        u_act = torch.maximum(torch.minimum(u_act, ep['shi']), -ep['slo'])   # несимметричное насыщение
        coupled = torch.bmm(ep['B'], u_act.unsqueeze(2)).squeeze(2)          # B u с перекрёстными связями
        self.state = (self.x_bar + ep['a'] * (self.state - self.x_bar) + ep['rate'][:, t]
                      - ep['c'] * coupled + ep['noise'][:, t - 1])           # формула (34) с рандомизацией
        self._observe()                                         # нормализация и eps(t+1)
        if not self.reward_on:                                  # оценка и учитель - без награды
            return None
        return self._reward(u, du, e_old)

    def _reward(self, u, du, e_old):
        rw, st, dt = RL['REWARD'], self.state, self.dt          # коэффициенты, состояние, точность
        e = self.E[:, self.t]                                   # средняя ошибка eps(t+1) по показателям
        ae = e.abs()                                            # |eps|
        self.held = torch.where(ae <= DELTA, self.held + 1, torch.zeros_like(self.held))   # счётчик удержания в допуске
        out = ((st > self.ucl) | (st < self.lcl)).to(dt).mean(1)                   # доля показателей вне коридора
        mag = ((torch.relu(st - self.ucl) + torch.relu(self.lcl - st)) / self.S).mean(1)   # глубина выхода, в долях H_i
        ez = self.N[:, self.t] - G                              # ошибки показателей N_i - g (без весов)
        zone = torch.relu(ez.abs() - DELTA).sum(1)              # суммарное выходящее за допуск отклонение зон
        r = (-rw['K_E'] * ae - rw['K_ZONE'] * zone - rw['K_OUT'] * out - rw['K_OUT_MAG'] * mag   # -|dK|, зоны, коридор
             - rw['LAMBDA_U'] * (u ** 2).mean(1) - rw['LAMBDA_DU'] * (du ** 2).mean(1)   # энергия и рывки
             + rw['BONUS'] * (self.held >= N_CONS).to(dt))      # бонус: |dK| <= delta уже T_h тактов подряд
        if RL['REF']['MODE'] == 'tau':                          # эталон: |e| сжимается не медленнее exp(-1/T_ref)
            r = r - rw['LAMBDA_REF'] * torch.relu(ae - self.rho * e_old.abs() - DELTA)   # штраф за отставание
            over = torch.relu(-torch.sign(self.e_peak) * e - self.os_ref * self.e_peak.abs() - DELTA)   # выброс за ноль сверх sigma_ref
            r = r - rw['LAMBDA_OS'] * over                      # штраф за перерегулирование
            self.e_peak = torch.where(self.held >= N_CONS, torch.zeros_like(e),
                                      torch.where(ae > self.e_peak.abs(), e, self.e_peak))   # пик (сброс после успокоения)
        else:                                                   # эталон: не хуже эталонного ПИД на том же объекте
            r = r - rw['LAMBDA_REF'] * torch.relu(ae - self.shadow.E[:, self.t].abs())
        return r.float()                                        # награда в float32 для буферов PPO


# ---------------------------- сеть регулятора (пространство поиска) ----------------------------
def _norm(kind, ch):
    """Нормализация признаков; BatchNorm в RL не используется (зависит от состава батча)."""
    if kind == 'layer':                                         # LayerNorm по (каналы, время)
        return nn.GroupNorm(1, ch)
    if kind == 'group':                                         # GroupNorm: 8 групп каналов (ширины кратны 8)
        return nn.GroupNorm(min(8, ch), ch)
    if kind == 'none':                                          # без нормализации
        return nn.Identity()
    raise ValueError(f"norm={kind!r}: в RL допустимы 'layer', 'group', 'none'")


class RLNet(nn.Module):
    """Свёртки по окну 8 x WINDOW -> (сжатие) -> [признаки, skip, z, c_ref] -> FC -> выход."""

    def __init__(self, hp, in_ch, extra_dim_, n_out):
        super().__init__()
        n = int(hp['n_conv'])                                   # число свёрточных слоёв
        chs = [in_ch] + [int(hp[f'c{i}']) for i in range(1, n + 1)]   # ширины слоёв
        ks = [int(hp[f'k{i}']) for i in range(1, n + 1)]        # длины фильтров (нечётные: длина окна сохраняется)
        self.convs = nn.ModuleList([nn.Conv1d(chs[i], chs[i + 1], ks[i], padding=ks[i] // 2) for i in range(n)])
        self.norms = nn.ModuleList([_norm(hp['norm'], chs[i + 1]) for i in range(n)])   # нормализация после свёртки
        c_last = chs[-1]                                        # каналов на выходе свёрток
        self.film = None                                        # FiLM: c_ref модулирует каналы последней свёртки
        if hp.get('film') and extra_dim_ > 0:
            self.film = nn.Linear(extra_dim_, 2 * c_last)       # gamma и beta на каждый канал
            nn.init.zeros_(self.film.weight)                    # старт с тождественного преобразования
            nn.init.zeros_(self.film.bias)
        self.pool = nn.AdaptiveAvgPool1d(int(hp['pool'])) if hp['temporal'] == 'avg' else None   # сжатие по времени
        t_len = int(hp['pool']) if self.pool is not None else WINDOW    # длина после сжатия (без сжатия - все 32 такта)
        self.skip = int(hp['skip'])                             # последних тактов входа - прямо в FC
        self.z = int(hp['ctx_z'])                               # размер контекста истории (0 - энкодер выключен)
        self.ctx_type = hp.get('ctx_type', 'conv')              # тип контекстного энкодера
        if self.z and self.ctx_type == 'gru':                   # GRU по истории (eps, N, u)
            self.ctx_rnn = nn.GRU(in_ch, 32, batch_first=True)
            self.ctx_out = nn.Linear(32, self.z)
        elif self.z:                                            # Conv1D по истории + усреднение по времени
            self.ctx_conv = nn.Sequential(nn.Conv1d(in_ch, 32, 5, padding=2), nn.ReLU(),
                                          nn.Conv1d(32, 32, 3, padding=1), nn.ReLU())
            self.ctx_out = nn.Linear(32, self.z)
        feat = c_last * t_len + in_ch * self.skip + self.z + extra_dim_   # размер входа FC
        self.fc1 = nn.Linear(feat, int(hp['fc']))               # полносвязный слой подбираемой ширины
        self.fc2 = nn.Linear(int(hp['fc']), n_out)              # выход: 8 (актор) или 1 (критик)
        self.relu = nn.ReLU()                                   # не inplace: тот же модуль используется в CUDA Graph

    def forward(self, x, extra):                                # x: (B, 8, WINDOW), extra: (B, E)
        h = x
        for conv, norm in zip(self.convs, self.norms):          # свёрточный стек
            h = self.relu(norm(conv(h)))
        if self.film is not None:                               # модуляция по c_ref
            g, b = self.film(extra).chunk(2, dim=1)
            h = h * (1.0 + g[:, :, None]) + b[:, :, None]
        if self.pool is not None:                               # сжатие по времени
            h = self.pool(h)
        parts = [h.flatten(1)]                                  # признаки свёрток
        if self.skip:                                           # skip-связь: последние такты входа как есть
            parts.append(x[:, :, -self.skip:].flatten(1))
        if self.z and self.ctx_type == 'gru':                   # контекст по GRU: последнее скрытое состояние
            _, hn = self.ctx_rnn(x.transpose(1, 2))
            parts.append(torch.tanh(self.ctx_out(hn[-1])))
        elif self.z:                                            # контекст по Conv1D
            parts.append(torch.tanh(self.ctx_out(self.ctx_conv(x).mean(dim=2))))
        if extra.shape[1]:                                      # c_ref (и u_ПИД) конкатенацией перед FC
            parts.append(extra)
        return self.fc2(self.relu(self.fc1(torch.cat(parts, dim=1))))


class ActorCritic(nn.Module):
    """Актор (подбираемая сеть) + критик той же архитектуры (видит параметры объекта) + log sigma."""

    def __init__(self, hp):
        super().__init__()
        self.hp, self.head = dict(hp), hp['head']               # архитектура и тип выхода
        e = extra_dim(self.head)                                # длина вектора мимо свёрток
        self.priv_critic = RL['PPO']['PRIV_CRITIC']             # асимметричный actor-critic
        self.actor = RLNet(hp, N_FEATURES, e, N_PARAMS)         # регулятор: 8 выходов
        self.critic = RLNet(hp, N_FEATURES, e + (PRIV_DIM if self.priv_critic else 0), 1)   # ценность состояния
        self.log_std = nn.Parameter(torch.full((N_PARAMS,), float(RL['PPO']['LOG_STD_INIT'])))   # разброс политики
        nn.init.orthogonal_(self.actor.fc2.weight, 0.01)        # малый выход в начале: u ~ 0 (или ~ ПИД)
        nn.init.zeros_(self.actor.fc2.bias)

    def value(self, x, extra, priv):
        z = torch.cat([extra, priv], dim=1) if self.priv_critic else extra   # критику - ещё и параметры объекта
        return self.critic(x, z).squeeze(1)                     # (B,)

    def n_actor_params(self):
        return sum(p.numel() for p in self.actor.parameters()) + self.log_std.numel()   # в регуляторе только актор


def soft_deadzone(u, threshold=None):
    """Мягкая мёртвая зона выходного сигнала: |u| <= threshold -> строго 0,
    иначе sign(u) * (|u| - threshold). Непрерывна в нуле и не пропускает шум малой амплитуды."""
    if threshold is None:                                       # порог по умолчанию - из конфигурации
        threshold = float(RL['HEAD'].get('DEADZONE', 0.1))
    if threshold <= 0.0:                                        # зона выключена
        return u
    return torch.sign(u) * torch.relu(u.abs() - threshold)      # «сжатие» к нулю вне зоны


def command(head, raw, u_prev, u_pid, deadzone=True, threshold=None):
    """Выход сети -> команда u в [-1; 1]: прямое u, приращение (интегральное действие) или поправка к ПИД.
    Мёртвая зона применяется к вкладу сети: у 'direct' - ко всему выходу, у 'delta'/'pid_res' - к приращению
    или поправке, чтобы не обнулять предыдущую команду и ПИД-базу в невозмущённых каналах."""
    h = RL['HEAD']                                              # пределы приращения и поправки
    if head == 'direct':                                        # прямое u
        u = torch.tanh(raw)
        return soft_deadzone(u, threshold) if deadzone else u
    if head == 'delta':                                         # u_t = clip(u_{t-1} + du)
        du = h['DU_MAX'] * torch.tanh(raw)
        if deadzone:
            du = soft_deadzone(du, threshold)
        return torch.clamp(u_prev + du, -1.0, 1.0)
    if head == 'pid_res':                                       # ПИД + ограниченная поправка
        res = h['RES_MAX'] * torch.tanh(raw)
        if deadzone:
            res = soft_deadzone(res, threshold)
        return torch.clamp(u_pid + res, -1.0, 1.0)
    raise ValueError(f'head={head!r}')


def gauss_logp(raw, mu, log_std):
    """log pi(raw) для диагонального нормального закона (без torch.distributions - совместимо с CUDA Graph)."""
    return (-0.5 * ((raw - mu) / log_std.exp()) ** 2 - log_std - 0.5 * LOG2PI).sum(-1)


# ---------------------------- регуляторы для оценки (единый интерфейс) ----------------------------
class RLPolicy:
    """Обученный актор, детерминированно (среднее политики)."""

    def __init__(self, ac, feat=None):
        self.ac, self.head, self.feat = ac.eval(), ac.head, feat or RL['FEAT']   # сеть, тип выхода, вид входа
        self.need_pid = self.head == 'pid_res'                  # среде нужен внутренний ПИД

    def reset(self, env):
        pass                                                    # состояния у сети нет: всё в окне среды

    def act(self, env):
        raw = self.ac.actor(env.window(self.feat), env.extra(self.head))   # среднее политики
        return command(self.head, raw.to(env.dt), env.u_last, env.u_pid)  # в команду привода


class PIDBaseline:
    """ПИД-эталон (тот же, что учитель) с номинальными коэффициентами."""
    need_pid = False                                            # ПИД свой, среде не нужен

    def reset(self, env):
        c = RL['PID']                                           # коэффициенты из config (или pid_int.json в Colab)
        self.pid = PIDTorch(env.B, env.dev, env.dt, c['KP'], c['KI'], c['KD'])

    def act(self, env):
        return self.pid.act(env.N[:, env.t])                    # по текущим N_i


class LegacyCNNPolicy:
    """CNN после имитации: на вход подаются те же 8 нормализованных показателей N_i, что и в CNNPolicy."""
    need_pid = False

    def __init__(self, model):
        self.model = model.eval()                               # BatchNorm - в режиме eval

    def reset(self, env):
        pass

    def act(self, env):
        return self.model(env.window()).to(env.dt)              # выход tanh


@torch.no_grad()
def run_policy(policy, ep, dev):
    """Прогон пакета эпизодов с регулятором; (E, U) остаются на устройстве."""
    env = RLEnv(dev, reward=False, pid_inner=policy.need_pid)   # среда без награды
    env.reset(ep)
    policy.reset(env)
    U = torch.zeros((env.B, T_TOTAL, N_PARAMS), dtype=env.dt, device=dev)   # команды по тактам
    for t in range(N_STEPS):                                    # весь эпизод на устройстве
        U[:, t] = policy.act(env)
        env.step(U[:, t])
    U[:, -1] = policy.act(env)                                  # последний такт - для полноты метрик (как Sim.run)
    return env.E, U, env.N


def cl_metrics(E, U, N, ep_np):
    """Метрики замкнутого контура по эпизодам: IAE, Ts, перерегулирование, энергия, ложные срабатывания, J."""
    m = RL['METRIC']                                            # веса и параметры метрики
    t0s, quiet, names = ep_np['t0'], ep_np['quiet'], ep_np['names']
    rows = []                                                   # метрики каждого эпизода
    for b in range(len(t0s)):
        t0 = int(t0s[b])
        e, u = E[b, t0:t0 + T_OBS], U[b, t0:t0 + T_OBS]         # окно наблюдения после возмущения
        ez = N[b, t0:t0 + T_OBS] - G                            # ошибки показателей на окне наблюдения
        n = len(e)                                              # для C1-C9 короче T_OBS (t0 = 60..80)
        if quiet[b]:                                            # без возмущения важны только ложные срабатывания
            rows.append({'name': names[b], 'quiet': True, 'FA': float(np.mean(np.abs(u).max(axis=1) > IDLE_THRESHOLD))})
            continue
        ok = np.abs(e) <= DELTA                                 # такты в допуске
        settled = (np.lib.stride_tricks.sliding_window_view(ok, N_CONS).all(axis=1)
                   if n >= N_CONS else np.zeros(1, dtype=bool))  # T_h тактов подряд в допуске
        sign = np.sign(e[1:6].mean())                           # знак начального отклонения
        os_ = max(0.0, float((-sign * e[m['OS_FROM']:]).max())) if sign != 0 and n > m['OS_FROM'] else 0.0   # как в pid_tune
        rows.append({'name': names[b], 'quiet': False,
                     'IAE': float(np.abs(e).sum()),             # интеграл модуля ошибки
                     'TS': float(settled.argmax()) if settled.any() else float(n),   # время переходного процесса
                     'OS': os_,                                 # перерегулирование
                     'EN': float((u ** 2).sum(axis=1).mean()),   # энергия управления на такт
                     'ZE': float(np.maximum(np.abs(ez) - DELTA, 0.0).sum(axis=1).mean())})   # отклонение зон вне допуска
    return _aggregate(rows)


def _aggregate(rows, by_name=True):
    """Средние метрики по эпизодам и комбинированная метрика J; by_name - то же отдельно по подписям."""
    m = RL['METRIC']                                            # веса метрики
    dist = [r for r in rows if not r['quiet']]                  # эпизоды с возмущением
    qt = [r for r in rows if r['quiet']]                        # эпизоды без возмущения
    mean = lambda rs, k: float(np.mean([r[k] for r in rs])) if rs else 0.0   # среднее (0, если эпизодов нет)
    o = {k: mean(dist, k) for k in ('IAE', 'TS', 'OS', 'EN', 'ZE')}   # качество переходного процесса
    o['FA'] = mean(qt, 'FA')                                    # доля тактов с ложным срабатыванием
    o['J'] = (m['W_IAE'] * o['IAE'] + m['W_TS'] * o['TS'] / T_OBS + m['W_OS'] * o['OS']
              + m['W_EN'] * o['EN'] + m['W_FA'] * o['FA'] + m['W_ZE'] * o['ZE'])   # комбинированная метрика (меньше - лучше)
    if by_name:                                                 # разбивка по ситуациям C1-C9 / объектам OOD
        o['by_name'] = {g: _aggregate([r for r in rows if r['name'] == g], False)
                        for g in dict.fromkeys(r['name'] for r in rows)}
    return o


def rl_evaluate(policy, ep, dev):
    """Оценка регулятора на выборке ep (уже на устройстве): одна пересылка результатов на CPU."""
    E, U, N = run_policy(policy, ep, dev)                       # весь контур на устройстве
    return cl_metrics(E.cpu().numpy(), U.cpu().numpy(), N.cpu().numpy(), ep['np'])


def make_suites(dev, which=('nominal', 'dr', 'ood', 'critical')):
    """Оценочные выборки на устройстве (строятся один раз)."""
    build = {'nominal': suite_nominal, 'dr': suite_dr, 'ood': suite_ood, 'critical': suite_critical}
    return {k: to_device(build[k](), dev) for k in which}


# ---------------------------- ограничения на сеть ----------------------------
def infer_ms(hp, reps=100):
    """Медианное время вывода актора на такт: CPU, батч 1, текущее число потоков."""
    ac = ActorCritic(hp).eval()                                 # отдельная копия на CPU
    x = torch.zeros((1, N_FEATURES, WINDOW))                    # одно окно
    e = torch.zeros((1, extra_dim(hp['head'])))                 # c_ref (+ u_ПИД)
    times = []
    with torch.inference_mode():
        for i in range(reps + 20):
            t = time.perf_counter()
            ac.actor(x, e)
            if i >= 20:                                         # первые 20 вызовов - прогрев
                times.append(time.perf_counter() - t)
    return 1000.0 * float(np.median(times))                     # мс


def arch_limits(hp):
    """(в пределах?, {params, size_mb, infer_ms}, превышение) - проверка до обучения."""
    lim = RL['SEARCH']                                          # пределы из конфигурации
    n_par = ActorCritic(hp).n_actor_params()                    # параметров регулятора
    info = {'params': int(n_par), 'size_mb': n_par * 4 / 2 ** 20, 'infer_ms': infer_ms(hp)}   # float32 = 4 байта
    excess = (max(0.0, info['params'] / lim['MAX_PARAMS'] - 1) + max(0.0, info['size_mb'] / lim['MAX_SIZE_MB'] - 1)
              + max(0.0, info['infer_ms'] / lim['MAX_INFER_MS'] - 1))   # суммарное относительное превышение
    return excess == 0.0, info, excess


# ---------------------------- PPO ----------------------------
class PPO:
    """
    PPO на пакете сред. Сбор опыта - на устройстве без пересылок по тактам; шаг обновления (выборка
    минибатча, прямой/обратный проход, обрезка градиента, Adam) на GPU записывается в CUDA Graph,
    как в train_model: индексы минибатчей и счётчик - тензоры, lr - тензор, состояние после прогрева откатывается.
    """

    def __init__(self, arch, train_hp, dev, seed, log=print):
        self.dev, self.cfg, self.log = dev, RL['PPO'], log      # устройство, настройки, вывод сообщений
        torch.manual_seed(seed)                                 # воспроизводимость: веса и шум действий от seed
        self.rng = np.random.RandomState(seed)                  # эпизоды обучения от того же seed
        self.ac = ActorCritic(arch).to(dev)                     # сеть
        self.head, self.feat = arch['head'], RL['FEAT']         # тип выхода и вид входа
        self.n_env = int(self.cfg['N_ENVS'].get(dev.type, 64))  # число параллельных сред
        self.env = RLEnv(dev, reward=True, pid_inner=self.head == 'pid_res')   # пакет сред (создаётся один раз)
        self.cuda = dev.type == 'cuda'
        lr0 = float(train_hp['LR'])
        self.params = list(self.ac.parameters())
        self.opt = optim.Adam(self.params, lr=torch.tensor(lr0, device=dev) if self.cuda else lr0,
                              weight_decay=float(train_hp['WEIGHT_DECAY']),
                              capturable=self.cuda, foreach=self.cuda or None)   # как в train_model (CUDA Graph)
        s, b, e = N_STEPS, self.n_env, extra_dim(self.head)     # такты, среды, длина extra
        z = lambda *shape: torch.zeros(shape, device=dev)       # буфер float32 на устройстве
        self.obs, self.ext, self.prv = z(s, b, N_FEATURES, WINDOW), z(s, b, e), z(s, b, PRIV_DIM)   # входы
        self.raw, self.logp, self.val = z(s, b, N_PARAMS), z(s, b), z(s, b)   # действия, log pi, ценности
        self.rew, self.adv, self.ret = z(s, b), z(s, b), z(s, b)              # награды, преимущества, цели критика
        n = s * b                                               # переходов за итерацию
        self.f = {'obs': self.obs.view(n, N_FEATURES, WINDOW), 'ext': self.ext.view(n, e),   # плоские виды тех же буферов
                  'prv': self.prv.view(n, PRIV_DIM), 'raw': self.raw.view(n, N_PARAMS),       # (адреса не меняются -
                  'logp': self.logp.view(n), 'adv': self.adv.view(n), 'ret': self.ret.view(n)}   # граф их переиспользует)
        self.mb = min(int(train_hp['BATCH_SIZE']), n)          # размер минибатча
        self.nb = n // self.mb                                  # минибатчей за эпоху (неполный хвост отбрасывается)
        self.idx2d = torch.randperm(n, device=dev)[:self.nb * self.mb].view(self.nb, self.mb).clone()   # индексы эпохи
        self.k = torch.zeros(1, dtype=torch.long, device=dev)   # номер минибатча (растёт внутри шага)
        self.pg_coef = torch.ones((), device=dev)               # 0 - обучается только критик (после имитации)
        self.kl_acc = torch.zeros((), device=dev)               # сумма KL за эпоху
        gru = int(arch.get('ctx_z', 0)) > 0 and arch.get('ctx_type') == 'gru'   # cuDNN RNN в графе не записываем
        self.use_graph = self.cuda and self.cfg['CUDA_GRAPH'] and not gru
        self.graph, self.iter, self.vf_warmup = None, 0, 0      # граф, номер итерации, итераций разогрева критика

    # ---- имитация ПИД-эталона (необязательный этап) ----
    def pretrain_imitation(self):
        cfg, dev = RL['IMIT'], self.dev                         # настройки имитации
        env = RLEnv(dev, reward=False, pid_inner=True)          # учитель - внутренний ПИД среды
        X, EX, UP, UPID, Y = [], [], [], [], []                 # входы, extra, u(t-1), u_ПИД, цель
        left = int(cfg['EPISODES'])                             # сколько эпизодов осталось собрать
        with torch.no_grad():
            while left > 0:
                nb = min(self.n_env, left)                      # эпизодов в этом пакете
                ep = to_device(sample_episodes(self.rng, nb, RL['DR'] if cfg['DR'] else RL['NOMINAL']), dev)
                env.reset(ep)
                for t in range(N_STEPS):                        # ПИД ведёт объект, сеть смотрит
                    X.append(env.window(self.feat))
                    EX.append(env.extra(self.head))
                    UP.append(env.u_last.float())
                    UPID.append(env.u_pid.float())
                    Y.append(env.u_pid.float())                 # цель - команда учителя
                    env.step(env.u_pid)
                left -= nb
        X, EX, UP, UPID, Y = (torch.cat(a) for a in (X, EX, UP, UPID, Y))   # выборка целиком на устройстве
        opt = optim.Adam(self.ac.actor.parameters(), lr=float(cfg['LR']))   # обучается только актор
        n, bs = len(X), int(cfg['BATCH'])
        for epoch in range(int(cfg['EPOCHS'])):
            perm = torch.randperm(n, device=dev)                # перемешивание
            total = torch.zeros((), device=dev)                 # сумма потерь без синхронизации в цикле
            for i in range(0, n - bs + 1, bs):
                idx = perm[i:i + bs]
                u = command(self.head, self.ac.actor(X[idx], EX[idx]), UP[idx], UPID[idx],
                            deadzone=False)                 # имитация учителя без мёртвой зоны (чистая супервизия)
                loss = weighted_mse(u, Y[idx])                  # та же функция потерь, что у прежней имитации
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.ac.actor.parameters(), 1.0)   # как в train_model
                opt.step()
                total += loss.detach()
            if epoch == 0 or (epoch + 1) % 5 == 0:
                self.log(f"  имитация: эпоха {epoch + 1}, loss = {total.item() / max(1, n // bs):.5f}")
        with torch.no_grad():
            self.ac.log_std.fill_(float(cfg['LOG_STD_AFTER']))  # узкая политика: не разрушить учителя шумом
        self.vf_warmup = int(cfg['VF_WARMUP_ITERS'])            # сначала доучить критика

    # ---- сбор опыта ----
    @torch.no_grad()
    def collect(self):
        env, ac, g, lam = self.env, self.ac, self.cfg['GAMMA'], self.cfg['LAMBDA']
        env.reset(to_device(sample_episodes(self.rng, self.n_env), self.dev))   # новые объекты и возмущения (одна пересылка)
        ac.eval()
        std = ac.log_std.exp()                                  # разброс политики
        for t in range(N_STEPS):                                # эпизод целиком на устройстве
            x, e, p = env.window(self.feat), env.extra(self.head), env.priv()
            mu = ac.actor(x, e)                                 # среднее политики
            raw = mu + std * torch.randn_like(mu)               # стохастическое действие (до tanh / clip)
            self.obs[t], self.ext[t], self.prv[t], self.raw[t] = x, e, p, raw   # в буферы
            self.logp[t] = gauss_logp(raw, mu, ac.log_std)      # log pi старой политики
            self.val[t] = ac.value(x, e, p)                     # ценность состояния
            self.rew[t] = env.step(command(self.head, raw.to(env.dt), env.u_last, env.u_pid))   # шаг всех сред
        last = ac.value(env.window(self.feat), env.extra(self.head), env.priv())   # эпизод обрезан по времени: бутстрэп
        a = torch.zeros(self.n_env, device=self.dev)            # GAE
        for t in reversed(range(N_STEPS)):
            nv = last if t == N_STEPS - 1 else self.val[t + 1]  # ценность следующего состояния
            a = self.rew[t] + g * nv - self.val[t] + g * lam * a
            self.adv[t] = a
        self.ret.copy_(self.adv + self.val)                     # цель критика
        return self.rew.sum(0).mean()                           # средний доход эпизода (тензор, без синхронизации)

    # ---- шаг обновления ----
    def _step(self):
        f, c = self.f, self.cfg
        idx = self.idx2d.index_select(0, self.k).view(-1)       # текущий минибатч (индекс - тензор, графу подходит)
        x, e, p = f['obs'].index_select(0, idx), f['ext'].index_select(0, idx), f['prv'].index_select(0, idx)
        raw, lp_old = f['raw'].index_select(0, idx), f['logp'].index_select(0, idx)
        adv, ret = f['adv'].index_select(0, idx), f['ret'].index_select(0, idx)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)           # нормировка преимуществ
        lp = gauss_logp(raw, self.ac.actor(x, e), self.ac.log_std)   # log pi новой политики
        ratio = torch.exp(lp - lp_old)                          # отношение вероятностей
        pg = -torch.min(ratio * adv, torch.clamp(ratio, 1 - c['CLIP'], 1 + c['CLIP']) * adv).mean()   # PPO-clip
        vf = ((self.ac.value(x, e, p) - ret) ** 2).mean()       # потери критика
        ent = (self.ac.log_std + 0.5 * (LOG2PI + 1.0)).sum()    # энтропия нормального закона
        loss = self.pg_coef * pg + c['VF_COEF'] * vf - c['ENT_COEF'] * ent
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, c['MAX_GRAD'])
        self.opt.step()
        self.kl_acc.add_((lp_old - lp).mean().detach())         # приближённый KL
        self.k.add_(1)                                          # к следующему минибатчу

    def _runner(self):
        if not self.use_graph:                                  # CPU / XPU / GRU - обычный шаг
            return self._step
        if self.graph is None:                                  # запись графа - один раз, как в train_model
            snap = {k: v.clone() for k, v in self.ac.state_dict().items()}   # веса до прогрева
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):                       # прогрев: 3 шага до записи графа
                for _ in range(3):
                    self._step()
            torch.cuda.current_stream().wait_stream(side)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):                  # запись одного шага
                self._step()
            with torch.no_grad():                               # откат прогрева: веса и состояние Adam
                for k, v in self.ac.state_dict().items():
                    v.copy_(snap[k])
                for st in self.opt.state.values():
                    for v in st.values():
                        v.zero_()
        return self.graph.replay                                # шаг = один вызов

    def update(self):
        run, n, kl = self._runner(), self.f['obs'].shape[0], 0.0
        for _ in range(int(self.cfg['EPOCHS'])):                # проходы по собранным данным
            self.idx2d.copy_(torch.randperm(n, device=self.dev)[:self.nb * self.mb].view(self.nb, self.mb))
            self.k.zero_()
            self.kl_acc.zero_()
            self.ac.train()
            for _ in range(self.nb):
                run()
            kl = self.kl_acc.item() / self.nb                   # единственная синхронизация за эпоху
            if kl > self.cfg['TARGET_KL']:                      # политика ушла слишком далеко - стоп
                break
        return kl

    def iterate(self):
        """Одна итерация: сбор N_ENVS эпизодов и обновление; возвращает средний доход и KL."""
        ret = self.collect()                                    # опыт
        self.pg_coef.fill_(0.0 if self.iter < self.vf_warmup else 1.0)   # разогрев критика после имитации
        kl = self.update()                                      # обучение
        self.iter += 1
        return {'ret': ret.item(), 'kl': kl}


# ---------------------------- отчёт ----------------------------
SUITE_TITLES = {'nominal': 'Номинальный объект', 'dr': 'Рандомизированные объекты (обучающие диапазоны)',
                'ood': 'Объекты вне обучающих диапазонов', 'critical': 'Критические ситуации C1-C9 (отложенная проверка)'}


def _ms(vals):
    """Среднее и разброс по seed."""
    return float(np.mean(vals)), float(np.std(vals)) if len(vals) > 1 else 0.0


def print_report(results, log=print):
    """results: {регулятор: {выборка: [метрики по seed]}} -> таблицы для каждой выборки."""
    keys = ('IAE', 'TS', 'OS', 'EN', 'ZE', 'FA', 'J')           # столбцы
    for suite, title in SUITE_TITLES.items():
        rows = [(n, r[suite]) for n, r in results.items() if suite in r]   # регуляторы, оценённые на этой выборке
        if not rows:
            continue
        log(f"\n--- {title} ---")
        log(f"{'регулятор':<28}" + ''.join(f"{k:>16}" for k in keys))
        for name, runs in rows:
            cells = []
            for k in keys:
                mu, sd = _ms([m[k] for m in runs])              # по seed
                cells.append(f"{mu:>9.3f}" + (f" ±{sd:<5.3f}" if sd else ' ' * 7))
            log(f"{name:<28}" + ''.join(cells))
        if suite in ('critical', 'ood'):                        # разбивка J по ситуациям / объектам
            names = list(rows[0][1][0].get('by_name', {}))
            if names:
                log(f"{'J по ' + ('ситуациям' if suite == 'critical' else 'объектам'):<28}"
                    + ''.join(f"{s[:10]:>11}" for s in names))
                for name, runs in rows:
                    log(f"{name:<28}" + ''.join(f"{_ms([m['by_name'][s]['J'] for m in runs])[0]:>11.3f}" for s in names))
# <<< END RL-CORE
