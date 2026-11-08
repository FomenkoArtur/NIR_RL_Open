"""
PID-регулятор для цифрового двойника участка металлообработки (WP5).
Работает с PlantODESystem: берёт уставку g(t) из модели, формирует
управляющее воздействие по формуле u = Kp*e + Ki*integral(e) + Kd*de/dt
с насыщением u_max и антивиндпом.
"""

import numpy as np


class PIDController:
    """Пропорционально-интегрально-дифференциальный регулятор.

    Управляющее воздействие вычисляется по отклонению состояния x от
    программной уставки g(t) (задаётся объектом PlantODESystem).
    """

    def __init__(self, plant, kp=1.0, ki=0.1, kd=0.05, u_max=1.0):
        # plant — объект PlantODESystem, задаёт уставку g(t) и размерность состояния
        self.plant = plant
        self.kp = float(kp)       # коэффициент пропорциональной части
        self.ki = float(ki)       # коэффициент интегральной части
        self.kd = float(kd)       # коэффициент дифференциальной части
        self.u_max = float(u_max) # предел насыщения управляющего воздействия
        self.sign = np.ones(8)    # знак обратной связи (отрицательная ОС)
        self.reset()

    def reset(self):
        """Сброс внутреннего состояния: интеграл ошибки и предыдущее значение."""
        self.integral = np.zeros(8)     # накопленная ошибка (интегральная часть)
        self.prev_error = None          # ошибка на предыдущем такте (для дифференциала)
        self.prev_t = None              # время предыдущего такта

    def act(self, x, t):
        """Вычислить управляющее воздействие u(x, t).

        Parameters
        ----------
        x : np.ndarray, shape (8,)
            Текущее состояние объекта (восемь показателей).
        t : float
            Текущее время; по нему определяется уставка g(t).

        Returns
        -------
        np.ndarray, shape (8,)
            Управляющее воздействие, ограниченное по модулю u_max.
        """
        # Ошибка регулирования: расхождение текущего состояния с программной уставкой
        g = self.plant.setpoint(t)
        error = x - g
        dt = 0.1  # такт управления, с

        # Дифференциальная часть: скорость изменения ошибки
        if self.prev_error is None:
            derivative = np.zeros_like(error)
        else:
            derivative = (error - self.prev_error) / dt

        # Формула PID: u = Kp*e + Ki*integral(e) + Kd*de/dt
        u_raw = self.sign * (self.kp * error + self.ki * self.integral + self.kd * derivative)
        u = np.clip(u_raw, -self.u_max, self.u_max)

        # Антивиндп: при насыщении привода не накапливаем интеграл,
        # если ошибка толкает привод дальше в насыщение
        saturated = np.abs(u_raw) >= self.u_max
        pushing = saturated & (error * u_raw > 0)
        self.integral = self.integral + (~pushing) * (error * dt)

        self.prev_error = error
        self.prev_t = t
        return u
