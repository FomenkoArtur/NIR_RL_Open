"""
pid_control.py — PID-регулятор для PlantODESystem (WP5).

Реализует обратную связь по каналам (одно управление u_i на каждый показатель x_i):

    u = Kp*e + Ki*integral(e) + Kd*de/dt,   e = x - g(t),

где g(t) — уставка программы обработки из PlantODESystem.setpoint(t).
Перед подачей на привод управление ограничивается (насыщение) по модулю:
    |u_i| <= u_max.

Применён признак действия каждого привода, вычисленный из диагонали
матрицы эффективности управления b_ii, чтобы обратная связь была
отрицательной независимо от знака b_ii (для некоторых каналов положительное
увеличение ускоряет нагрев или нарастание силы, поэтому знак уравновешивается
коэффициентом s_i). Также есть защита от накопления интеграла при насыщении
привода (anti-windup).

Только numpy, без torch.
"""

from __future__ import annotations

import numpy as np


class PIDController:
    """ПИД-регулятор с разделением по восьми каналам объекта."""

    def __init__(self, plant, kp=1.0, ki=0.1, kd=0.05, u_max=1.0):
        """
        Parameters
        ----------
        plant : PlantODESystem
            Объект модели: задаёт уставку setpoint(t), матрицу управления b
            и постоянные времени tau.
        kp, ki, kd : float
            Коэффициенты пропорционального, интегрального и дифференциального
            звеньев регулятора.
        u_max : float
            Предел насыщения управляющего воздействия по каждому каналу.
        """
        self.plant = plant
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.u_max = float(u_max)

        # s_i = +1, если положительный u_i уменьшает x_i (b_ii < 0),
        # s_i = -1, если положительный u_i увеличивает x_i (b_ii > 0).
        # Домножение ошибки на s_i делает обратную связь отрицательной для
        # любого знака коэффициента эффективности b_ii.
        diag_b = np.diag(np.asarray(plant.b, dtype=float))
        self.sign = np.where(diag_b > 0.0, -1.0, 1.0)

        self.reset()

    def reset(self):
        """Сброс внутреннего состояния: интеграл ошибки и предыдущее отклонение."""
        self.integral = np.zeros(self.plant.N_STATES, dtype=float)
        self.prev_error = None
        self.prev_t = None

    def act(self, x, t):
        """
        Расчёт управляющего воздействия на текущем такте.

        Parameters
        ----------
        x : array-like, shape (8,)
            Текущее состояние объекта.
        t : float
            Текущее время (нужно для уставки g(t) и шага dt).

        Returns
        -------
        u : np.ndarray, shape (8,)
            Насыщенное управляющее воздействие.
        """
        x = np.asarray(x, dtype=float).reshape(self.plant.N_STATES)
        t = float(t)

        # Уставка программы обработки на момент t (разд. «уставка g(t)»).
        g = np.asarray(self.plant.setpoint(t), dtype=float).reshape(self.plant.N_STATES)
        error = x - g  # ошибка отслеживания e_i = x_i - g_i

        # Шаг по времени для интеграла и дифференциала.
        if self.prev_t is None:
            dt = 0.1
        else:
            dt = t - self.prev_t
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 0.1
        dt = max(dt, 1e-6)

        # Дифференциальная составляющая: de/dt (производная ошибки по времени).
        if self.prev_error is None:
            derivative = np.zeros_like(error)
        else:
            derivative = (error - self.prev_error) / dt

        # Формула ПИД: u = Kp*e + Ki*∫e + Kd*de/dt с учётом признака привода s_i.
        proportional = self.kp * error
        integral_term = self.ki * self.integral
        derivative_term = self.kd * derivative
        u_raw = self.sign * (proportional + integral_term + derivative_term)

        # Насыщение привода: ограничиваем управление по модулю u_max.
        u = np.clip(u_raw, -self.u_max, self.u_max)

        # Anti-windup: не накапливаем интеграл, когда привод уже насыщен и
        # ошибка всё ещё толкает управление в сторону насыщения — иначе
        # интеграл раздувается и выводит систему из равновесия.
        saturated = np.abs(u_raw) >= self.u_max
        pushing_into_sat = saturated & (error * u_raw > 0.0)
        self.integral = self.integral + (~pushing_into_sat) * (error * dt)

        self.prev_error = error
        self.prev_t = t

        return u.astype(float)
