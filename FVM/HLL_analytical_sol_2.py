import numpy as np
from scipy import constants
import matplotlib.pyplot as plt
def calculate_test_solution_y2(Nx, Nvar, tol, Efield, x_cell, Te, vBohm, velo_0, ndens0):
    """
    Calculate test solution `test_sol`.

    Parameters:
        Nx (int): Number of cells.
        Nvar (int): Number of variables.
        tol (float): Tolerance for convergence.
        Efield (float): Electric field constant.
        x_cell (numpy.ndarray): Cell positions (1D array of size Nx+1).
        Te (float): Electron temperature.
        vBohm (float): Bohm velocity.
        velo_0 (float): Reference velocity.
        ndens0 (float): Reference number density.

    Returns:
        numpy.ndarray: Test solution array of shape (Nx+2, Nvar).
    """
    # Initialize arrays
    yy = np.ones(Nx + 2, dtype=np.float64)
    test_sol = np.zeros((Nx + 2, Nvar), dtype=np.float64)

    for i in range(1, Nx + 2):
        yy_old = yy[i]
        # print(yy_old)
        while True:
            yy_new = (2 * (Efield * x_cell[i - 1] / Te + np.log(yy_old)) * (vBohm**2) / (velo_0**2) + 1.0) / yy_old
            if abs((yy_new - yy_old) / yy_new) < tol:
                # print("----", ndens0 / yy_new, velo_0 * yy_new)
                print("----",  yy_new, x_cell[i - 1] )

                yy[i] = yy_new
                break
            yy_old = yy_new

    # Boundary condition
    yy[0] = 1.0

    # Calculate test_sol
    test_sol[:, 0] = ndens0 / yy
    test_sol[:, 1] = velo_0 * yy

    return test_sol

# Example usage (replace with actual parameters):
Nx = 1000
Nvar = 2
tol = 1e-9
Efield = 300.0
x_cell = np.linspace(0, 0.1, Nx + 1)
Te = 2.0
mi = 131.3 * constants.atomic_mass
vBohm = np.sqrt(constants.elementary_charge * 2.0 / mi)
velo_0 = 2.0 * vBohm
ndens0 = 1.0e+09


result = calculate_test_solution_y2(Nx, Nvar, tol, Efield, x_cell, Te, vBohm, velo_0, ndens0)
plt.plot(x_cell, (result[0:- 1, 1] / velo_0), 'r--' )
plt.plot(x_cell, (result[0:- 1, 0] / ndens0), 'k--' )
plt.yscale('log')
plt.show()

print((result[0:- 1, 0] / ndens0))