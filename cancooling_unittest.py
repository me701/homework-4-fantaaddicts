import unittest
import numpy as np
from HW4cancooling import run_simulation, h_spin, h_wait

class TempTest(unittest.TestCase):

    def test_temp_decrease(self):
        T_field, _ = run_simulation(h_spin, h_wait)
        N_z, N_r = T_field.shape
        for j in range(N_z):  
            for i in range(N_r - 1):
                with self.subTest(z_index=j, r_index=i):
                    self.assertLessEqual(
                        T_field[j, i+1],
                        T_field[j, i],
                        f"Temperature did not decrease radially at z={j}, r={i}"
                    )

if __name__ == "__main__":
    unittest.main()