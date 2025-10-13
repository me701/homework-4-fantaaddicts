import numpy as np
import matplotlib.pyplot as plt
from starter_code import construct_system
import time as ostime
import os
import imageio

# Geometry & Discretization
R_can, H_can = 0.03, 0.12
N_r, N_z = 15, 30
N_total = N_r * N_z

# --- Thermal Properties (Water) with UNCERTAINTY ---
T_data = np.array([273.15, 275, 280, 290, 295, 300, 305, 310])
k_data = np.array([560, 560, 568, 590, 598, 607, 616, 624]) * 1e-3

# Baseline k function
k_coeffs = np.polyfit(T_data, k_data, 1)
k_of_T_base = np.poly1d(k_coeffs)

# Calculate uncertainty in k
k_predicted = k_of_T_base(T_data)
percent_error = np.max(np.abs(k_data - k_predicted) / k_data)
k_uncertainty = percent_error

# Baseline properties
rho = 1000
cp_base = 4186

# Temperatures
T_initial, T_inf, T_target = 308.0, 273.15, 280.0

# Baseline Convection coefficients
h_spin_base, h_wait_base = 1160.0, 380.0
h_uncertainty = 0.10 # 10%

# Time stepping & Grid
dt = 2.0
r_edge = np.linspace(0, R_can, N_r + 1)
z_edge = np.linspace(0, H_can, N_z + 1)
r_center = (r_edge[1:] + r_edge[:-1]) / 2
z_center = (z_edge[1:] + z_edge[:-1]) / 2
qppp = lambda r, z: 0.0

def calculate_cooling_time(h_s, h_w, cp, k_func):
    """
    Runs a silent simulation for a given set of parameters and returns the time.
    """
    T = np.full(N_total, T_initial)
    t = 0.0
    center_idx = 0 + (N_z // 2) * N_r
    T_center = T[center_idx]
    
    while T_center > T_target:
        # Determine the phase and h_val
        h_val = 0
        if t <= 30: h_val = h_s
        elif t <= 330: h_val = h_w
        elif t <= 360: h_val = h_s
        elif t <= 660: h_val = h_w
        elif t <= 690: h_val = h_s
        else: h_val = h_w
        
        K_grid = k_func(T).reshape((N_z, N_r))
        kfun_local = lambda r, z: K_grid[np.abs(z_center - z).argmin(), np.abs(r_center - r).argmin()]
        
        A, b = construct_system(r_center, z_center, kfun_local, qppp, ('neumann', 0.0), ('neumann', 0.0), ('neumann', 0.0))
        
        dr, dz = r_center[1] - r_center[0], z_center[1] - z_center[0]
        for j in range(N_z):
            for i in range(N_r):
                p = i + j * N_r
                if j == N_z - 1 or j == 0:
                    A[p, p] += h_val / dz
                    b[p] += (h_val / dz) * T_inf
                if i == N_r - 1:
                    ri, re = r_center[i], r_center[i] + dr / 2.0
                    factor = re / (ri * dr)
                    A[p, p] += h_val * factor
                    b[p] += (h_val * factor) * T_inf
        
        M_dt = rho * cp / dt
        A_implicit = M_dt * np.identity(N_total) + A
        b_implicit = M_dt * T + b
        T = np.linalg.solve(A_implicit, b_implicit)
        
        t += dt
        T_center = T[center_idx]
        
    return t

# Simulate CI
print("--- Calculating 95% Confidence Interval for Cooling Time ---")

# Slowest Cooling
print("\nRunning Worst-Case Simulation (Upper Bound)...")
k_lower_bound_func = np.poly1d(k_coeffs * (1 - k_uncertainty))
time_upper = calculate_cooling_time(
    h_s=h_spin_base * (1 - h_uncertainty),
    h_w=h_wait_base * (1 - h_uncertainty),
    cp=cp_base * (1 + 0),
    k_func=k_lower_bound_func
)

# Fastest Cooling
print("Running Best-Case Simulation (Lower Bound)...")
k_upper_bound_func = np.poly1d(k_coeffs * (1 + k_uncertainty))
time_lower = calculate_cooling_time(
    h_s=h_spin_base * (1 + h_uncertainty),
    h_w=h_wait_base * (1 + h_uncertainty),
    cp=cp_base * (1 - 0),
    k_func=k_upper_bound_func
)

# Regular Simulation
print("Running Baseline Simulation...")
time_baseline = calculate_cooling_time(h_spin_base, h_wait_base, cp_base, k_of_T_base)

print("\n--- ANALYSIS COMPLETE ---")
print(f"Uncertainty in k fit found to be: {k_uncertainty*100:.2f}%")
print(f"\nBaseline estimated cooling time: {time_baseline:.1f} seconds ({time_baseline/60:.2f} minutes)")
print(f"The 95% confidence interval for the cooling time is [{time_lower:.1f}, {time_upper:.1f}] seconds.")
print(f"This corresponds to a range of [{time_lower/60:.2f}, {time_upper/60:.2f}] minutes.")

# Get the animation
def run_and_animate_baseline(h_s, h_w, cp, k_func):
    animation_dir = "animation_frames"
    if not os.path.exists(animation_dir):
        os.makedirs(animation_dir)
    filenames = []
    
    T = np.full(N_total, T_initial)
    t = 0.0
    step = 0
    center_idx = 0 + (N_z // 2) * N_r
    T_center = T[center_idx]
    
    print("\n--- Starting Simulation for Animation (with variable k) ---")
    start_time = ostime.time()
    
    while T_center > T_target:
        phase = ""
        h_val = 0
        if t <= 30: phase, h_val = "Spin 1", h_s
        elif t <= 330: phase, h_val = "Wait 1", h_w
        elif t <= 360: phase, h_val = "Spin 2", h_s
        elif t <= 660: phase, h_val = "Wait 2", h_w
        elif t <= 690: phase, h_val = "Spin 3", h_s
        else: phase, h_val = "Final Wait", h_w
        
        K_grid = k_func(T).reshape((N_z, N_r))
        kfun_local = lambda r, z: K_grid[np.abs(z_center - z).argmin(), np.abs(r_center - r).argmin()]
        A, b = construct_system(r_center, z_center, kfun_local, qppp, ('neumann', 0.0), ('neumann', 0.0), ('neumann', 0.0))
        
        dr, dz = r_center[1] - r_center[0], z_center[1] - z_center[0]
        for j in range(N_z):
            for i in range(N_r):
                p = i + j * N_r
                if j == N_z - 1 or j == 0:
                    A[p, p] += h_val / dz
                    b[p] += (h_val / dz) * T_inf
                if i == N_r - 1:
                    ri, re = r_center[i], r_center[i] + dr / 2.0
                    factor = re / (ri * dr)
                    A[p, p] += h_val * factor
                    b[p] += (h_val * factor) * T_inf
        
        M_dt = rho * cp / dt
        A_implicit = M_dt * np.identity(N_total) + A
        b_implicit = M_dt * T + b
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
            plt.savefig(filename)
            plt.close()
            filenames.append(filename)
    
    print(f"\nAnimation simulation finished in {ostime.time() - start_time:.2f} seconds.")
    return filenames

image_filenames = run_and_animate_baseline(h_spin_base, h_wait_base, cp_base, k_of_T_base)

print("\n--- Creating GIF Animation ---")
with imageio.get_writer('can_cooling_animation_final.gif', mode='I', duration=0.1) as writer:
    for filename in image_filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
print("Animation saved as 'can_cooling_animation_final.gif'")

print("--- Cleaning up frame files ---")
for filename in image_filenames:
    os.remove(filename)
os.rmdir("animation_frames")
print("Cleanup complete.")