import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# --- 1. Data Extraction from the Table (Restricted to T=273.15 K to T=310 K) ---
# Only including data points from 273.15 K up to and including 310 K.

# T (K)
T_data = np.array([
    273.15, 275, 280, 290, 295, 300, 305, 310
])

# Specific Heat (Cp, Saturated Liquid, kJ/kg·K)
Cp_data = np.array([
    4.217, 4.217, 4.208, 4.195, 4.191, 4.181, 4.178, 4.178
])

# Viscosity (mu, Saturated Liquid, N·s/m^2)
# Data is given as mu * 10^3, so we multiply by 1e-3
mu_x1000_data = np.array([
    1.791, 1.675, 1.442, 1.139, 1.010, 0.898, 0.803, 0.720
])
mu_data = mu_x1000_data * 1e-3

# Thermal Conductivity (k, Saturated Liquid, W/m·K)
# Data is given as k * 10^3, so we multiply by 1e-3
k_x1000_data = np.array([
    560, 560, 568, 590, 598, 607, 616, 624
])
k_data = k_x1000_data * 1e-3

# Prandtl Number (Pr, Saturated Liquid)
Pr_data = np.array([
    13.2, 12.0, 9.6, 8.0, 7.4, 6.62, 5.72, 5.2
])


# --- 2. Defining the Polynomial Function for Curve Fitting ---
# A 3rd degree polynomial: f(T) = a*T^3 + b*T^2 + c*T + d
def poly3(T, a, b, c, d):
    return a * T**3 + b * T**2 + c * T + d


# --- 3. Function to Perform Fit and Print Results ---
def perform_fit(T, P_data, P_name, P_units):
    try:
        # Perform the curve fit
        # Note: Since we only have 8 data points, a 3rd degree polynomial (4 coefficients) 
        # is a reasonable fit, but higher degrees would lead to overfitting.
        popt, pcov = curve_fit(poly3, T, P_data)
        a, b, c, d = popt

        print(f"\n--- Fitted Function for {P_name} ---")
        print(f"Function: {P_name}(T) = a*T^3 + b*T^2 + c*T + d (Units: {P_units})")
        print(f"Coefficients (a, b, c, d):")
        print(f"a = {a:e}")
        print(f"b = {b:e}")
        print(f"c = {c:e}")
        print(f"d = {d:e}")
        print("-" * 40)

        # Return coefficients for plotting
        return popt, f'{P_name} ({P_units})'

    except RuntimeError:
        print(f"\n! Error: Curve fit for {P_name} failed. Data may be too non-linear for a 3rd-degree polynomial.")
        return None, None

# --- 4. Perform Fits for all four properties ---

# List to store results for plotting
plot_data = []

# 1. Viscosity (mu)
coeffs_mu, label_mu = perform_fit(T_data, mu_data, "Viscosity", "N·s/m²")
if coeffs_mu is not None:
    plot_data.append((mu_data, coeffs_mu, label_mu))

# 2. Thermal Conductivity (k)
coeffs_k, label_k = perform_fit(T_data, k_data, "Thermal Conductivity", "W/m·K")
if coeffs_k is not None:
    plot_data.append((k_data, coeffs_k, label_k))

# 3. Specific Heat (Cp)
coeffs_Cp, label_Cp = perform_fit(T_data, Cp_data, "Specific Heat (Cp)", "kJ/kg·K")
if coeffs_Cp is not None:
    plot_data.append((Cp_data, coeffs_Cp, label_Cp))

# 4. Prandtl Number (Pr)
coeffs_Pr, label_Pr = perform_fit(T_data, Pr_data, "Prandtl Number", "Dimensionless")
if coeffs_Pr is not None:
    plot_data.append((Pr_data, coeffs_Pr, label_Pr))


# --- 5. Plotting the Results ---

T_fit = np.linspace(T_data.min(), T_data.max(), 500)

plt.figure(figsize=(12, 12))

for i, (P_data, coeffs, label) in enumerate(plot_data):
    # Calculate the fitted curve values
    P_fit = poly3(T_fit, *coeffs)

    plt.subplot(2, 2, i + 1) # Create 2x2 grid of plots
    plt.plot(T_data, P_data, 'o', label='Original Data', color='red')
    plt.plot(T_fit, P_fit, '-', label=f'Fitted 3rd Degree Polynomial', color='blue')
    plt.xlabel('Temperature, T (K)')
    plt.ylabel(label)
    plt.title(f'Curve Fit for {label.split(" (")[0]} (Range: 273 K - 310 K)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

plt.show()

# Final summary of functions for easy copy-paste
print("\n" + "="*50)
print("FINAL CURVE FIT FUNCTIONS (T in Kelvin, Range: 273 K - 310 K):")
print("="*50)

if coeffs_mu is not None:
    a, b, c, d = coeffs_mu
    print(f"1. Viscosity (mu, N·s/m²):")
    print(f"mu(T) = ({a:e})*T^3 + ({b:e})*T^2 + ({c:e})*T + ({d:e})")

if coeffs_k is not None:
    a, b, c, d = coeffs_k
    print(f"\n2. Thermal Conductivity (k, W/m·K):")
    print(f"k(T) = ({a:e})*T^3 + ({b:e})*T^2 + ({c:e})*T + ({d:e})")

if coeffs_Cp is not None:
    a, b, c, d = coeffs_Cp
    print(f"\n3. Specific Heat (Cp, kJ/kg·K):")
    print(f"Cp(T) = ({a:e})*T^3 + ({b:e})*T^2 + ({c:e})*T + ({d:e})")

if coeffs_Pr is not None:
    a, b, c, d = coeffs_Pr
    print(f"\n4. Prandtl Number (Pr, Dimensionless):")
    print(f"Pr(T) = ({a:e})*T^3 + ({b:e})*T^2 + ({c:e})*T + ({d:e})")
