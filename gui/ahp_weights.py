"""Расчёт весов МАИ методом анализа иерархии (Саати) — только для вывода.

Модуль не участвует в обучении регулятора (суперкритерий K в обучении не
используется, см. запрет в TASK.md). Он нужен только для отображения
весов частных критериев на вкладках и графиках GUI.

Реализация по Саати:
    1. Нормализация столбцов парной матрицы сравнений, среднее по строкам -> вектор приоритетов w.
    2. lambda_max = mean((A w)_i / w_i), CI = (lambda_max - n) / (n - 1).
    3. CR = CI / RI[n], где RI[n] - таблица случайных индексов Саати.
    4. Согласованно, если CR < 0.10.
"""
from __future__ import annotations

import numpy as np

from config import T_START, T_OBS

# Таблица средних случайных индексов RI (Шкала Саати), n = 1..15.
_RI = {
    1: 0.00, 2: 0.00, 3: 0.52, 4: 0.89, 5: 1.11,
    6: 1.25, 7: 1.35, 8: 1.40, 9: 1.45, 10: 1.49,
    11: 1.52, 12: 1.54, 13: 1.56, 14: 1.58, 15: 1.59,
}

# Максимально допустимое значение отношения согласованности Saaty.
CR_LIMIT = 0.10


def ahp_weights(matrix, t=None):
    """Приоритетные веса критериев методом парных сравнений Саати.

    Parameters
    ----------
    matrix : array-like, shape (n, n)
        Парная матрица сравнений критериев (шкала 1..9; a_ij = важность i
        относительно j, a_ji = 1/a_ij).
    t : float, optional
        Время (такт профиля обработки). Принимается для совместимости с
        временными профилями (см. weights_at), в самой функции не используется.

    Returns
    -------
    w : np.ndarray, shape (n,)
        Вектор весов приоритетов (сумма равна 1).
    ci : float
        Индекс согласованности Saaty, CI = (lambda_max - n) / (n - 1).
    cr : float
        Отношение согласованности, CR = CI / RI[n].
    lam_max : float
        Оценка наибольшего собственного числа парной матрицы.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("парная матрица должна быть квадратной")
    n = matrix.shape[0]

    col_sum = matrix.sum(axis=0)
    if np.any(col_sum <= 0):
        raise ValueError("элементы парной матрицы должны быть положительными")
    norm = matrix / col_sum[None, :]
    w = norm.mean(axis=1)
    w = w / w.sum()

    Aw = matrix @ w
    lam_max = float((Aw / w).mean()) if np.all(w > 0) else float(n)
    ci = (lam_max - n) / (n - 1) if n > 1 else 0.0
    ri = float(_RI.get(n, _RI[max(_RI)]))
    cr = ci / ri if ri > 0 else 0.0
    return w, ci, cr, lam_max


def is_consistent(matrix):
    """True, если матрица парных сравнений согласована (CR < 0.10)."""
    _, _, cr, _ = ahp_weights(matrix)
    return bool(cr < CR_LIMIT)


def _stage_matrix(stage):
    """Парная матрица сравнений критериев для одного этапа профиля обработки.

    На каждом этапе приоритеты разные: на подготовке важнее геометрические
    показатели (x1 диаметр, x4 размер), на основной обработке — температурный
    режим (x7) и охлаждение (x8), на завершении — вибрация (x3, x6).
    """
    n = 8
    m = np.ones((n, n))
    if stage == "prep":
        important = {0, 3}  # диаметр и размер
        for i in important:
            for j in range(n):
                if i != j:
                    m[i, j] = 3.0
        for j in important:
            for i in range(n):
                if i != j:
                    m[i, j] = 1.0 / 3.0
    elif stage == "proc":
        thermal = {6, 7}   # температура и охлаждение
        surface = {0, 1, 2, 3, 4, 5}
        for i in thermal:
            for j in surface:
                m[i, j] = 3.0
        for i in surface:
            for j in thermal:
                m[i, j] = 1.0 / 3.0
    elif stage == "finish":
        for i in (2, 5):   # вибрация в зоне резания
            for j in range(n):
                if i != j:
                    m[i, j] = 2.0
    else:
        raise ValueError(f"неизвестный этап профиля обработки: {stage!r}")
    return m


def _stage_weights(stage, n_params):
    """Приоритетные веса для одного этапа через ahp_weights."""
    matrix = _stage_matrix(stage)
    if matrix.shape[0] != n_params:
        full = np.ones((n_params, n_params))
        k = min(n_params, matrix.shape[0])
        full[:k, :k] = matrix[:k, :k]
        matrix = full
    w, _, _, _ = ahp_weights(matrix)
    return w


def weights_at(t, n_params=8):
    """Вектор весов МАИ, зависящий от времени t (профиль обработки).

    Профиль обработки делится на этапы:
        t < T_START - 10                      : подготовка (подбор режима),
        T_START - 10 .. T_START + T_OBS + 10  : основная обработка,
        t > T_START + T_OBS + 10              : завершение (отживление/закалка).

    На границах этапов веса интерполируются линейно (окно 20 тактов), чтобы
    не было скачков при переключении этапа. Возвращает вектор длины n_params
    (сумма равна 1).
    """
    t = float(t)
    n = int(n_params)
    t_prep_end = T_START - 10.0
    t_proc_end = T_START + T_OBS + 10.0
    blend = 20.0

    if t <= t_prep_end:
        return _stage_weights("prep", n)
    if t >= t_proc_end:
        return _stage_weights("finish", n)

    w = _stage_weights("proc", n)
    s_in = min(max((t - t_prep_end) / blend, 0.0), 1.0)
    if s_in < 1.0:
        w = (1.0 - s_in) * _stage_weights("prep", n) + s_in * w
    s_out = min(max((t - (t_proc_end - blend)) / blend, 0.0), 1.0)
    if s_out > 0.0:
        w = (1.0 - s_out) * w + s_out * _stage_weights("finish", n)

    w = np.asarray(w, dtype=float)
    return w / w.sum()


def ahp_report(matrix):
    """Словарь с результатами расчёта весов Саати (для вывода в GUI)."""
    w, ci, cr, lam_max = ahp_weights(matrix)
    return {
        "weights": w,
        "lambda_max": lam_max,
        "CI": ci,
        "CR": cr,
        "consistent": bool(cr < CR_LIMIT),
    }
