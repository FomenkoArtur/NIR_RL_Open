import numpy as np

class PIDController:
    def __init__(self, plant=None, kp=1.0, ki=0.1, kd=0.05, u_max=1.0):
        self.kp = float(kp); self.ki = float(ki); self.kd = float(kd); self.u_max = float(u_max)
        self.sign = np.ones(8); self.reset()
    def reset(self):
        self.integral = np.zeros(8); self.prev_error = None
    def act(self, error, t):
        dt = 0.1
        derivative = np.zeros_like(error) if self.prev_error is None else (error - self.prev_error) / dt
        u_raw = self.sign * (self.kp * error + self.ki * self.integral + self.kd * derivative)
        u = np.clip(u_raw, -self.u_max, self.u_max)
        saturated = np.abs(u_raw) >= self.u_max
        pushing = saturated & (error * u_raw > 0)
        self.integral = self.integral + (~pushing) * (error * dt)
        self.prev_error = error
        return u
