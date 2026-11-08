"""
MPC-регулятор (модель預скающего управления) для цифрового двойника участка
металлообработки (WP5). Работает с PlantODESystem: берёт уставку g(t) из
модели, на горизонте horizon минимизирует Q*|x-g|^2 + R*|u|^2.
"""

import numpy as np
from scipy.optimize import minimize


class MPCController:
    """Регулятор модели預скающего управления.

    На каждом такте решает задачу ограниченной квадратичной оптимизации:
    минимизирует сумму квадратов отклонения состояния от уставки и
    затрат управления на горизонте horizon шагов.
    """

    def __init__(self, plant, horizon=10, Q=1.0, R=0.1, u_max=1.0):
        # plant — объект PlantODESystem, задаёт уставку g(t) и размерность состояния
        self.plant = plant
        self.horizon = int(horizon)   # горизонт предсказания, тактов
        self.Q = float(Q)             # вес отклонения от уставки в целевой функции
        self.R = float(R)             # вес затрат управления в целевой функции
        self.u_max = float(u_max)     # предел насыщения управляющего воздействия
        self.n = 8                    # размерность состояния и управления
        self.reset()

    def reset(self):
        """Сброс внутренней истории управления."""
        self.prev_u = None

    def act(self, x, t):
        """Вычислить управляющее воздействие u(x, t) методом MPC.

        Parameters
        ----------
        x : np.ndarray, shape (8,)
            Текущее состояние объекта.
        t : float
            Текущее время; по нему определяется уставка g(t).

        Returns
        -------
        np.ndarray, shape (8,)
            Первое управление из оптимальной последовательности на горизонте.
        """
        # Уставка на текущий момент времени (программное воздействие g(t))
        g = self.plant.setpoint(t)
        x_bar = self.plant.initial_state()  # центр состояния как база линеаризации
        u0 = np.zeros(self.horizon * self.n)

        # Простая линейная модель объекта для предсказания на горизонте:
        # x_{k+1} = x_bar + a*(x_k - x_bar) + b*u_k, где a ~ 0.99 (инерция),
        # b ~ -0.1 (управление возвращает к уставке)
        a_coef = 0.99
        b_coef = -0.10

        def cost(u_seq):
            u_seq = u_seq.reshape(self.horizon, self.n)
            cost_val = 0.0
            x_pred = x.copy()
            for k in range(self.horizon):
                # Шаг предсказания состояния по линейной модели
                x_pred = x_bar + a_coef * (x_pred - x_bar) + b_coef * u_seq[k]
                # Ошибка предсказанного состояния относительно уставки g(t)
                error = x_pred - g
                # Целевая функция: Q*|x-g|^2 + R*|u|^2
                cost_val += self.Q * float(np.sum(error ** 2)) + self.R * float(np.sum(u_seq[k] ** 2))
            return cost_val

        # Ограничения на управление: |u_i| <= u_max
        bounds = [(-self.u_max, self.u_max)] * (self.horizon * self.n)
        res = minimize(cost, u0, bounds=bounds, method='L-BFGS-B',
                       options={'maxiter': 20, 'ftol': 1e-9})
        return res.x[:self.n]
