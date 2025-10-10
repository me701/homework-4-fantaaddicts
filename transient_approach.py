# Trying out the transient approach

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from calculateh import calculate_h, cp_J

# can dimensions (mm)
D = 0.06
h = 0.12
Asurf = 2*np.pi*(D/2)*h + 2*np.pi*(D/2)**2
V = np.pi*(D/2)**2*h

# boundary conditions (K)
T_hotcar = 308.15
T_oo = 273.15
T_target = 280.372

# material properties
rho_water = 1000 # kg/m3

# transient lumped parameter equation
def T_prime(t, T_bar, T_oo):
    cp = cp_J(T_bar)
    h = calculate_h(T_bar, 'natural')
    return -(h*Asurf / (rho_water*cp*V))*(T_bar-T_oo)

# times of interest
t_initial = 0
t_final = 1000
t_eval = np.linspace(t_initial,t_final, 1000)


sol = solve_ivp(T_prime, [t_initial, t_final], [T_hotcar],
                args = (T_oo,),
                t_eval=t_eval)

plt.plot(sol.t, sol.y[0], "k")
plt.show()