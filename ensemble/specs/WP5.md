# WP5: PID и MPC на той же модели, честное сравнение

## Назначение
Создать отдельные регуляторы PID и MPC, работающие с PlantODESystem из WP1.
Сравнение в experiment.py на тех же сценариях, что у RL.

## Ссылки на работу
2.2.1, 3.3.

## Файлы
- pid_control.py — класс PIDController
- mpc_control.py — класс MPCController

## Требуемый API
class PIDController:
    def __init__(self, plant, kp=1.0, ki=0.1, kd=0.05, u_max=1.0): ...
    def reset(self): ...
    def act(self, x, t): ...

class MPCController:
    def __init__(self, plant, horizon=10, Q=1.0, R=0.1, u_max=1.0): ...
    def reset(self): ...
    def act(self, x, t): ...

## Требования
1. Оба класса принимают объект PlantODESystem в __init__.
2. act(x, t) возвращает np.ndarray формы (8,) — управляющее воздействие.
3. reset() сбрасывает внутреннее состояние интегратора/истории.
4. PID использует формулу u = Kp*e + Ki*integral(e) + Kd*de/dt с насыщением u_max.
5. MPC использует горизонт horizon и минимизирует Q*|x-g|^2 + R*|u|^2.
6. Используют setpoint(t) из PlantODESystem.
7. Комментарии с физическим смыслом.

## Запреты
- Не менять plant_equations.py, scenarios.py, rl_core.py.
- Не импортировать torch (только numpy).
- Не подгонять под тесты.

## Критерии приёмки
См. ensemble/acceptance/test_wp5.py.
