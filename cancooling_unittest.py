import unittest
import numpy as np

from HW4cancooling import (
    N_total, T_initial, T_inf, T_target, N_r, N_z, r_center, z_center, dt,
    get_h_values, construct_system, k_of_T_base, cp_of_T_base, qppp, rho_water
)

class TestFullSimulation(unittest.TestCase):

    def test_center_temperature_decreases_monotonically_throughout(self):
        """
        Runs the entire baseline simulation from start to finish and verifies
        that the center temperature is always decreasing or staying the same
        at every time step.
        """
        print("\n--- Running unittest ---")
        
        T = np.full(N_total, T_initial)
        t = 0.0
        center_idx = 0 + (N_z // 2) * N_r
        T_center = T[center_idx]
        
        center_temp_history = [T_center]
        
        while T_center > T_target:
            T_surface_avg = np.mean(np.concatenate([T[N_r-1::N_r], T[-N_r:]]))
            h_natural, h_forced = get_h_values(T_surface_avg, rpm=60)
            
            h_val = 0
            if t <= 30: h_val = h_forced
            elif t <= 330: h_val = h_natural
            elif t <= 360: h_val = h_forced
            elif t <= 660: h_val = h_natural
            elif t <= 690: h_val = h_forced
            else: h_val = h_natural

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
            T_new = np.linalg.solve(A_implicit, b_implicit)
            
            T = T_new
            t += dt
            T_center = T[center_idx]
            
            center_temp_history.append(T_center)

        for i in range(1, len(center_temp_history)):
            prev_temp = center_temp_history[i-1]
            curr_temp = center_temp_history[i]
            self.assertLessEqual(curr_temp, prev_temp, 
                                 f"Temperature INCORRECTLY INCREASED at step {i} (time {i*dt:.1f}s): {prev_temp:.4f} K -> {curr_temp:.4f} K")
        
        print(f"Test Passed: Center temperature decreased monotonically over {len(center_temp_history)-1} steps.")

# This makes the test runnable from the command line
if __name__ == '__main__':
    unittest.main()