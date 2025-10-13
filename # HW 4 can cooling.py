import numpy as np
import matplotlib.pyplot as plt
from starter_code import construct_system
import time as ostime
import os
import imageio

#Geometry
R_can = 0.03  # m
H_can = 0.12  # m

N_r = 15
N_z = 30
N_total = N_r * N_z

# --- Thermal Properties (Water) ---
# k values from the book (see latex write up for what book)
T_data = np.array([273.15, 275, 280, 290, 295, 300, 305, 310])
k_data = np.array([560, 560, 568, 590, 598, 607, 616, 624]) * 1e-3

# poly fit bc I know how to do this :)
k_coeffs = np.polyfit(T_data, k_data, 1)
k_of_T = np.poly1d(k_coeffs)

rho = 1000
cp = 4186

# temperatures
T_initial = 308.0
T_inf = 273.15
T_target = 280.0

# Convection coefficients (W/m^2.K)
h_spin = 1160.0
h_wait = 380.0

# time steps
dt = 2.0

r_edge = np.linspace(0, R_can, N_r + 1)
z_edge = np.linspace(0, H_can, N_z + 1)
r_center = (r_edge[1:] + r_edge[:-1]) / 2
z_center = (z_edge[1:] + z_edge[:-1]) / 2

# no heat production
qppp = lambda r, z: 0.0


def run_simulation(h_s, h_w):
    """Runs one full simulation, rebuilding the matrix at each step."""
    animation_dir = "animation_frames"
    if not os.path.exists(animation_dir):
        os.makedirs(animation_dir)
    filenames = []
    
    T = np.full(N_total, T_initial)
    t = 0.0
    current_h = -1
    step = 0
    center_idx = 0 + (N_z // 2) * N_r
    T_center = T[center_idx]
    
    print("--- Starting Simulation for Animation (with variable k) ---")
    start_time = ostime.time()
    
    while T_center > T_target:
        phase = ""
        h_val = 0
        if t <= 30:
            phase = "Spin 1"
            h_val = h_s
        elif t <= 330:
            phase = "Wait 1"
            h_val = h_w
        elif t <= 360:
            phase = "Spin 2"
            h_val = h_s
        elif t <= 660:
            phase = "Wait 2"
            h_val = h_w
        elif t <= 690:
            phase = "Spin 3"
            h_val = h_s        
        else:
            phase = "Final Wait"
            h_val = h_w
            
        # 1. Create a 2D field of k values based on the current temperature field T
        k_field = k_of_T(T.reshape((N_z, N_r)))
        
        # 2. Define a kfun that looks up the value from our pre-calculated field
        kfun = lambda r, z, i, j: k_field[j, i]
        
        # 3. Build the system using this new kfun
        K_grid = k_of_T(T).reshape((N_z, N_r))
        
        def construct_system_with_temp(r_points, z_points, k_temperature_func, qppp,
                                       BC_z, BC_r_top, Bc_r_bottom, temp_field):
            
            # Create a K-field based on the temperature from the previous timestep
            K_field = k_temperature_func(temp_field).reshape((N_z, N_r))
            
            # look up the needed k value
            kfun_local = lambda r, z: K_field[np.abs(z_points - z).argmin(), np.abs(r_points - r).argmin()]

            return construct_system(r_points, z_points, kfun_local, qppp, BC_z, BC_r_top, Bc_r_bottom)

        A, b = construct_system_with_temp(r_center, z_center, k_of_T, qppp,
                                                ('neumann', 0.0), ('neumann', 0.0), ('neumann', 0.0), T)
        
        # 4. Robin BC
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
        
        # do this for each time step
        M_dt = rho * cp / dt
        A_implicit = M_dt * np.identity(N_total) + A
        b_implicit = M_dt * T + b
        T = np.linalg.solve(A_implicit, b_implicit)
        
        t += dt
        step += 1
        T_center = T[center_idx]
        
        if step % 25 == 0:
            T_min, T_max = np.min(T), np.max(T)
            print(f"Time: {t:5.1f} s | Center Temp: {T_center:6.2f} K | Min Temp: {T_min:6.2f} K | Max Temp: {T_max:6.2f} K")
            
            plt.figure(figsize=(6, 10))
            im = plt.imshow(np.flipud(T.reshape((N_z, N_r))), cmap="inferno",
                            extent=[0, R_can * 100, 0, H_can * 100], aspect='auto',
                            vmin=273, vmax=T_initial)
            
            plt.colorbar(im, label="Temperature (K)")
            plt.title(f"Can Temperature at t = {t:.1f} s\nPhase: {phase}")
            plt.xlabel("Radius (cm)")
            plt.ylabel("Height (cm)")
            plt.tight_layout()
            
            filename = f"{animation_dir}/frame_{step//25:04d}.png"
            plt.savefig(filename)
            plt.close()
            filenames.append(filename)
            
    end_time = ostime.time()
    print("\n--- Simulation Complete ---")
    print(f"Total cooling time: {t:.1f} seconds and {t/60:.3f} min")
    print(f"Real time for simulation: {end_time - start_time:.2f} seconds")
    
    return filenames

#RUN SIM
image_filenames = run_simulation(h_spin, h_wait)

print("\n--- Creating GIF Animation ---")
with imageio.get_writer('can_cooling_animation_variable_k.gif', mode='I', duration=0.1) as writer:
    for filename in image_filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
print("Animation saved as 'can_cooling_animation_variable_k.gif'")

print("--- Cleaning up frame files ---")
for filename in image_filenames:
    os.remove(filename)
os.rmdir("animation_frames")
print("Cleanup complete.")

# Plot this
plt.plot
