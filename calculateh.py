import numpy as np
import matplotlib.pyplot as plt
# Constants
T_inf = 273.15  # Ice bath temperature in Kelvin (32 F)
rho = 1000.0    # Density (kg/m^3)
D = 0.06        # Diameter (m)
H = 0.12        # Height (m)
g = 9.81        # Gravity (m/s²)
beta = 0.0002   # Volumetric expansion coeff. for water
v_spin = 2*np.pi*0.03    # Spin Velocity (m/s)

# T dependent material properties functions
# Viscosity (mu, N·s/m²)
mu = lambda T: (-1.298928e-08)*T**3 + (1.184029e-05)*T**2 + (-3.616306e-03)*T + (3.708810e-01)
# Thermal Conductivity (k, W/m·K)
k = lambda T: (-7.417913e-07)*T**3 + (6.483221e-04)*T**2 + (-1.868641e-01)*T + (1.834640e+01)
# Prandtl Number (Pr, Dimensionless)
Pr = lambda T: (-3.028567e-04)*T**3 + (2.692449e-01)*T**2 + (-7.988231e+01)*T + (7.916496e+03)
# Specific Heat (Cp, J/kg·K) - Converted from kJ/kg·K
cp_J = lambda T: (((9.390326e-07)*T**3 + (-8.082996e-04)*T**2 + (2.304558e-01)*T + (-1.756103e+01)) * 1000)

def calculate_h(T_bar, convection_type):
    """
    Calculates the convective heat transfer coefficient (h).

    Args:
        T_bar (float): The current temperature of the beverage (K).
        convection_type (str): 'forced' for spinning, 'natural' for sitting.

    Returns:
        float: The calculated h value (W/m²·K).
    """
    # Film Temperature
    T_s = T_inf # Ice bath temperature
    T_f = (T_bar + T_s) / 2 
    
    # properties at T_f
    k_f = k(T_f)
    mu_f = mu(T_f)
    Pr_f = Pr(T_f)
    
    if convection_type == 'forced':
        Lc = D
        Re = (rho * v_spin * Lc) / mu_f
        
        if 2300 < Re < 1e5 and 0.48 < Pr_f < 592:
            Nu = 0.015 * (Re**0.83) * (Pr_f**0.42)
        else:
            Nu = 0.027 * (Re**0.8) * (Pr_f**(1/3))
        
        h = (Nu * k_f) / Lc
        return h

    elif convection_type == 'natural':
        Lc = D 
        
        # Rayleigh Number (Ra = Gr * Pr)
        Gr = (g * beta * (T_bar - T_inf) * (Lc**3)) / (mu_f / rho)**2
        Ra = Gr * Pr_f
        
        if 1e5 <= Ra <= 1e12:
            term1 = 0.387 * (Ra**(1/6))
            term2 = (1 + (0.559 / Pr_f)**(9/16))**(8/27)
            Nu = (0.6 + term1 / term2)**2
        else:
            Nu = 0.53 * (Ra**(1/4))
        
        h = (Nu * k_f) / Lc
        return h
        
    return 0

T_max = 308
T_min = 273

T_bar = np.linspace(308, 273, 100)

h_forced = np.array([calculate_h(T, 'forced') for T in T_bar])
h_natural = np.array([calculate_h(T, 'natural') for T in T_bar])

# Plotting setup
plt.figure(figsize=(10, 5))
plt.plot(T_bar, h_forced, label='Forced Convection (Spinning)', linewidth=2)
plt.plot(T_bar, h_natural, label='Natural Convection (Sitting)', linewidth=2, linestyle='--')

plt.xlabel('Beverage Temperature (K)')
plt.ylabel('Heat Transfer Coefficient, h (W/m²·K)')
plt.title('Heat Transfer Coefficient (h) as a Function of Temperature')
plt.legend()
plt.grid(True)

plt.show()
