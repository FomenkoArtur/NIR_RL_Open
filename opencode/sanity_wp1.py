import numpy as np
import plant_equations as pe

p = pe.PlantODESystem()
print("initial_state:", p.initial_state())
print("tolerance:", p.tolerance())
for t in (0.0, 20.0, 50.0, 80.0, 120.0):
    print("setpoint(%g) =" % t, p.setpoint(t))
x_star, u_star = p.equilibrium()
print("equilibrium:", x_star, u_star)
x = p.initial_state()
u = np.zeros(8)
for k in range(140):
    x = p.step(x, u, 0.1, k * 0.1)
print("after 140 steps:", x, "max|x| =", np.abs(x).max())
x2 = p.step(p.initial_state(), np.ones(8) * 0.5, 0.1, 0.0)
print("with u=0.5:", x2)
print("derivatives:", p.derivatives(p.initial_state(), np.zeros(8), 0.0))
