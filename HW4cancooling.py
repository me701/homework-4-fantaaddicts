import numpy as np
import matplotlib.pyplot as plt
from starter_code import construct_system
import time as ostime
import os
import imageio

# 1. Value setup

# Geometry & Discretization
R_can, H_can = 0.03, 0.12
D_can = 2 * R_can
N_r, N_z = 15, 30
N_total = N_r * N_z

# Temperatures
T_initial, T_inf, T_target = 308.0, 273.15, 280.0

# Temp dep  properties of water
T_prop_data = np.array([273.15, 275, 280, 290, 295, 300, 305, 310])
k_fluid_data = np.array([0.560, 0.560, 0.568, 0.590, 0.598, 0.607, 0.616, 0.624])
cp_fluid_data = np.array([4217, 4217, 4208, 4195, 4191, 4181, 4178, 4178])
mu_fluid_data = np.array([1.791, 1.675, 1.442, 1.139, 1.010, 0.898, 0.803, 0.720]) * 1e-3

# Create polynomial fits for each property
k_fluid_func = np.poly1d(np.polyfit(T_prop_data, k_fluid_data, 2))
cp_fluid_func = np.poly1d(np.polyfit(T_prop_data, cp_fluid_data, 2))
mu_fluid_func = np.poly1d(np.polyfit(T_prop_data, mu_fluid_data, 2))

# Other properties and constants
rho_water = 1000
beta = 0.0004
g = 9.81

# Thermal Properties 
k_can_data = np.array([560, 560, 568, 590, 598, 607, 616, 624]) * 1e-3
k_coeffs = np.polyfit(T_prop_data, k_can_data, 1)
k_of_T_base = np.poly1d(k_coeffs)

cp_can_data = np.array([4217, 4217, 4208, 4195, 4191, 4181, 4178, 4178])
cp_coeffs = np.polyfit(T_prop_data, cp_can_data, 2)
cp_of_T_base = np.poly1d(cp_coeffs)

# Calculate uncertainties from fits
k_predicted = k_of_T_base(T_prop_data)
k_uncertainty = np.max(np.abs(k_can_data - k_predicted) / k_can_data)

cp_predicted = cp_of_T_base(T_prop_data)
cp_uncertainty = np.max(np.abs(cp_can_data - cp_predicted) / cp_can_data)

h_uncertainty = 0.10 # 10% uncertainty assumed 

# Time stepping & Grid 
dt = 2.0
r_edge, z_edge = np.linspace(0, R_can, N_r + 1), np.linspace(0, H_can, N_z + 1)
r_center, z_center = (r_edge[1:] + r_edge[:-1]) / 2, (z_edge[1:] + z_edge[:-1]) / 2
qppp = lambda r, z: 0.0

# 2. h value calculation

def get_h_values(T_surface_avg, rpm):
    """Calculates h_natural and h_forced based on the current surface temp."""
    T_film = (T_surface_avg + T_inf) / 2.0
    
    k_f, cp_f, mu_f = k_fluid_func(T_film), cp_fluid_func(T_film), mu_fluid_func(T_film)
    Pr_f = (cp_f * mu_f) / k_f
    nu_f = mu_f / rho_water
    
    Gr = (g * beta * abs(T_surface_avg - T_inf) * H_can**3) / (nu_f**2)
    Ra = Gr * Pr_f
    Nu_natural = (0.6 + (0.387 * Ra**(1/6)) / (1 + (0.559 / Pr_f)**(9/16))**(8/27))**2
    h_natural = Nu_natural * k_f / H_can
    
    omega = rpm * (2 * np.pi / 60)
    velocity = omega * R_can
    Re = (rho_water * velocity * D_can) / mu_f
    
    if 2300 < Re < 1e5 and 0.48 < Pr_f < 592:
        Nu_forced = 0.015 * Re**0.85 * Pr_f**0.42
    else:
        Nu_forced = 0.027 * Re**0.8 * Pr_f**(1/3)
    
    h_forced = Nu_forced * k_f / D_can
    return h_natural, h_forced

# 3. Simulation

def calculate_cooling_time(rpm, cp_func_factor, k_func_factor, h_factor):
    """Runs the can cooling simulation with uncertainty factors."""
    T = np.full(N_total, T_initial)
    t = 0.0
    center_idx = 0 + (N_z // 2) * N_r
    T_center = T[center_idx]
    
    cp_func = np.poly1d(cp_coeffs * cp_func_factor)
    k_func = np.poly1d(k_coeffs * k_func_factor)

    while T_center > T_target:
        T_surface_avg = np.mean(np.concatenate([T[N_r-1::N_r], T[-N_r:]]))
        h_natural_base, h_forced_base = get_h_values(T_surface_avg, rpm)
        h_natural, h_forced = h_natural_base * h_factor, h_forced_base * h_factor
        
        h_val = 0
        if t <= 30: h_val = h_forced
        elif t <= 330: h_val = h_natural
        elif t <= 360: h_val = h_forced
        elif t <= 660: h_val = h_natural
        elif t <= 690: h_val = h_forced
        else: h_val = h_natural
        
        K_grid = k_func(T).reshape((N_z, N_r))
        kfun_local = lambda r, z: K_grid[np.abs(z_center - z).argmin(), np.abs(r_center - r).argmin()]
        A, b = construct_system(r_center, z_center, kfun_local, qppp, ('neumann', 0.0), ('neumann', 0.0), ('neumann', 0.0))
        
        dr, dz = r_center[1] - r_center[0], z_center[1] - z_center[0]
        for j in range(N_z):
            for i in range(N_r):
                p = i + j * N_r
                if j == N_z - 1 or j == 0: A[p, p] += h_val / dz; b[p] += (h_val / dz) * T_inf
                if i == N_r - 1:
                    factor = (r_center[i] + dr / 2.0) / (r_center[i] * dr)
                    A[p, p] += h_val * factor; b[p] += (h_val * factor) * T_inf
        
        cp_field = cp_func(T)
        M_dt_diag = rho_water * cp_field / dt
        A_implicit = np.diag(M_dt_diag) + A
        b_implicit = M_dt_diag * T + b
        T = np.linalg.solve(A_implicit, b_implicit)
        
        t += dt
        T_center = T[center_idx]
    return t

# We didn't know how to do an animation so the oracle helped :)
def run_and_animate_baseline(rpm):
    """Runs the can cooling simulation and saves animation frames."""
    animation_dir = "animation_frames"
    if not os.path.exists(animation_dir): os.makedirs(animation_dir)
    filenames = []
    
    T = np.full(N_total, T_initial)
    t = 0.0
    step = 0
    center_idx = 0 + (N_z // 2) * N_r
    T_center = T[center_idx]
    
    print("\n--- Starting Simulation for Animation ---")
    start_time = ostime.time()
    
    while T_center > T_target:
        T_surface_avg = np.mean(np.concatenate([T[N_r-1::N_r], T[-N_r:]]))
        h_natural, h_forced = get_h_values(T_surface_avg, rpm)
        
        phase, h_val = "", 0
        if t <= 30: phase, h_val = "Spin 1", h_forced
        elif t <= 330: phase, h_val = "Wait 1", h_natural
        elif t <= 360: phase, h_val = "Spin 2", h_forced
        elif t <= 660: phase, h_val = "Wait 2", h_natural
        elif t <= 690: phase, h_val = "Spin 3", h_forced
        else: phase, h_val = "Final Wait", h_natural
        
        K_grid = k_of_T_base(T).reshape((N_z, N_r))
        kfun_local = lambda r, z: K_grid[np.abs(z_center - z).argmin(), np.abs(r_center - r).argmin()]
        A, b = construct_system(r_center, z_center, kfun_local, qppp, ('neumann', 0.0), ('neumann', 0.0), ('neumann', 0.0))
        
        dr, dz = r_center[1] - r_center[0], z_center[1] - z_center[0]
        for j in range(N_z):
            for i in range(N_r):
                p = i + j * N_r
                if j == N_z - 1 or j == 0: A[p, p] += h_val / dz; b[p] += (h_val / dz) * T_inf
                if i == N_r - 1:
                    factor = (r_center[i] + dr / 2.0) / (r_center[i] * dr)
                    A[p, p] += h_val * factor; b[p] += (h_val * factor) * T_inf
        
        cp_field = cp_of_T_base(T)
        M_dt_diag = rho_water * cp_field / dt
        A_implicit = np.diag(M_dt_diag) + A
        b_implicit = M_dt_diag * T + b
        T = np.linalg.solve(A_implicit, b_implicit)
        t += dt
        step += 1
        T_center = T[center_idx]
        
        if step % 25 == 0:
            print(f"Time: {t:5.1f} s | Center Temp: {T_center:6.2f} K")
            plt.figure(figsize=(6, 10))
            im = plt.imshow(np.flipud(T.reshape((N_z, N_r))), cmap="inferno", extent=[0, R_can * 100, 0, H_can * 100], aspect='auto', vmin=273, vmax=T_initial)
            plt.colorbar(im, label="Temperature (K)")
            plt.title(f"Can Temperature at t = {t:.1f} s\nPhase: {phase}")
            plt.xlabel("Radius (cm)"), plt.ylabel("Height (cm)")
            plt.tight_layout()
            filename = f"{animation_dir}/frame_{step//25:04d}.png"
            plt.savefig(filename), plt.close()
            filenames.append(filename)
            
    print(f"\nAnimation simulation finished in {ostime.time() - start_time:.2f} seconds.")
    return filenames

# outputs
print("--- Calculating 95% Confidence Interval for Cooling Time ---")

# Worst-Case (Slowest)
time_upper = calculate_cooling_time(rpm=60, cp_func_factor=1+cp_uncertainty, k_func_factor=1-k_uncertainty, h_factor=1-h_uncertainty)
# Best-Case (Fastest)
time_lower = calculate_cooling_time(rpm=60, cp_func_factor=1-cp_uncertainty, k_func_factor=1+k_uncertainty, h_factor=1+h_uncertainty)
# Baseline
time_baseline = calculate_cooling_time(rpm=60, cp_func_factor=1, k_func_factor=1, h_factor=1)

print("\n--- ANALYSIS COMPLETE ---")
print(f"Uncertainty in k fit found to be: {k_uncertainty*100:.2f}%")
print(f"Uncertainty in Cp fit found to be: {cp_uncertainty*100:.2f}%")
print(f"\nBaseline estimated cooling time: {time_baseline:.1f} seconds ({time_baseline/60:.2f} minutes)")
print(f"The 95% confidence interval for the cooling time is [{time_lower:.1f}, {time_upper:.1f}] seconds.")
print(f"This corresponds to a range of [{time_lower/60:.2f}, {time_upper/60:.2f}] minutes.")

# --- Generate the Animation ---
image_filenames = run_and_animate_baseline(rpm=60)

print("\n--- Creating GIF Animation ---")
with imageio.get_writer('can_cooling_animation.gif', mode='I', duration=0.1) as writer:
    for filename in image_filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
print("Animation saved as 'can_cooling_animation.gif'")

print("--- Cleaning up frame files ---")
for filename in image_filenames:
    os.remove(filename)
os.rmdir("animation_frames")
print("Cleanup complete.")