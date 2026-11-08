# Калибровка: 2-3 ОДУ объекта управления

## Задача
Написать `plant_equations.py` с классом `PlantODESystem`, содержащим 2 ОДУ
(отклонение диаметра при точении x1 и шероховатость Ra x2), уставка g(t) = константа.

## API
class PlantODESystem:
    N_STATES = 2
    def __init__(self, params=None): ...
    def initial_state(self) -> np.ndarray: ...
    def derivatives(self, x, u, t) -> np.ndarray: ...
    def step(self, x, u, dt, t, noise=None, disturbance=None) -> np.ndarray: ...

## Критерий
Тест в `test_cal.py` зелёный.