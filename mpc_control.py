"""
mpc_control.py — Регулятор модели предсказуемого управления (MPC)
для PlantODESystem (WP5).

На каждом такте формируется прогноз состояния на горизонт предсказания и
решается задача оптимизации

    min_u  Q * sum_{k=0}^{H-1} |x_k - g(t + k*dt)|^2 + R * H * |u|^2

при модели 예측 x_{k+1} = x_k + dt * f(x_k, u, t_k), где f — правая часть
ОДЕ PlantODESystem.derivatives, и ограничении |u_i| <= u_max (насыщение).

Управление считается постоянным на всём горизонте (односъёмная модель), тогда
задача сводится к квадратичной и решается аналитически через решение системы
линейных уравнений. Ядро прогноза — правая часть ОДЕ, матрица эффективности
управления b и постоянные времени tau. Только numpy, без torch.
"""

from __future__ import annotations

import numpy as np


class MPCController:
    """Модель предсказуемого управления с квадратичной функцией потерь."""

    def __init__(self, plant, horizon=10, Q=1.0, R=0.1, u_max=1.0):
        """
        Parameters
        ----------
        plant : PlantODESystem
            Объект модели: задаёт уставку setpoint(t), динамику derivatives,
            матрицу управления b и постоянные времени tau.
        horizon : int
            Горизонт предсказания (число шагов вперёд).
        Q : float
            Коэффициент при ошибке отслеживания |x - g|^2.
        R : float
            Коэффициент при затратах на управление |u|^2.
        u_max : float
            Предел насыщения управляющего воздействия по каналу.
        """
        self.plant = plant
        self.horizon = int(horizon)
        self.Q = float(Q)
        self.R = float(R)
        self.u_max = float(u_max)

        # Матрица эффективности управления в виде dx/dt = a(x, t) + G u:
        # G_ij = b_ij / tau_i, то есть вклад управления u_j в производную
        # состояния x_i. Является ядром прогноза состояния на горизонте.
        tau = np.asarray(plant.tau, dtype=float)
        b = np.asarray(plant.b, dtype=float)
        self.G = b / tau[:, None]

        self.reset()

    def reset(self):
        """Сброс истории времени (очистка предыдущего момента t)."""
        self.prev_t = None

    def act(self, x, t):
        """
        Расчёт управления методом предсказуемого управления.

        Parameters
        ----------
        x : array-like, shape (8,)
            Текущее состояние объекта.
        t : float
            Текущее время.

        Returns
        -------
        u : np.ndarray, shape (8,)
            Оптимальное (в смысле функции потерь) насыщенное управление.
        """
        x = np.asarray(x, dtype=float).reshape(self.plant.N_STATES)
        t = float(t)

        # Шаг предсказания по времени.
        if self.prev_t is None:
            dt = 0.1
        else:
            dt = t - self.prev_t
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 0.1
        dt = max(dt, 1e-6)

        # Прогноз уставок на горизонте: g_k = g(t + k*dt), k = 0..H-1.
        # Используем setpoint(t) из PlantODESystem для каждого шага прогноза.
        ks = np.arange(self.horizon, dtype=float)
        t_grid = t + ks * dt
        g_seq = np.asarray(self.plant.setpoint(t_grid), dtype=float)
        if g_seq.ndim == 1:
            g_seq = np.tile(g_seq, (self.horizon, 1))  # (H, 8)

        # Дрейф объекта при нулевом управлении: a(x, t) = f(x, 0, t).
        # Замораживаем на текущем состоянии — линеаризация вокруг x.
        a = np.asarray(
            self.plant.derivatives(x, np.zeros(self.plant.N_STATES), t),
            dtype=float,
        )

        # Отклонение от уставки на шаге k при нулевом управлении:
        # x_k = x + k*dt*a  =>  r_k = x - g_k + k*dt*a.
        r = x[None, :] - g_seq + (ks * dt)[:, None] * a[None, :]  # (H, 8)

        # Квадратичная форма функции потерь:
        # J(u) = Q * sum_k |r_k + k*dt*G u|^2 + H*R*|u|^2 = u^T A u + 2 b^T u + const.
        s2 = float(np.sum((ks * dt) ** 2))            # sum_k (k*dt)^2
        A = self.Q * s2 * (self.G.T @ self.G) + (self.horizon * self.R) * np.eye(
            self.plant.N_STATES
        )                                             # матрица коэффициентов квадрата
        bvec = self.Q * (self.G.T @ np.sum((ks * dt)[:, None] * r, axis=0))  # линейный член

        # Аналитический минимизатор квадратичной функции потерь над горизонтом.
        try:
            u_star = -np.linalg.solve(A, bvec)
        except np.linalg.LinAlgError:
            u_star = -np.linalg.pinv(A) @ bvec

        # Насыщение приводов: ограничение управления по модулю u_max.
        u = np.clip(u_star, -self.u_max, self.u_max)

        if not np.all(np.isfinite(u)):
            u = np.zeros(self.plant.N_STATES, dtype=float)

        self.prev_t = t
        return u.astype(float)
