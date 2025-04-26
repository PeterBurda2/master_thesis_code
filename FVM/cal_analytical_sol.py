import numpy as np
from scipy import constants
import matplotlib.pyplot as plt


def fixed_point_method(x0, epsilon, n_iter, E, Te, vb, v0, x):

    n = 0 
    y_old = x0
    while True:
        y_new  =  ((2.0 * ((E / Te) * x + np.log(y_old))) * (vb/v0) ** 2 + 1) / y_old
        
        if np.abs((y_new - y_old) / y_old) < epsilon:
            # print(f"Solution: {y_new} at {x} with {n} iterations")
            # print(f"Solution: {y_new} at {x}")

            return y_new
        
        n += 1
        y_old = y_new

    # print(f"Maximum interations exceeded!")

def calc_analytical_solution(plot_bool:bool, N:int):

    
    x_cell = np.linspace(0.0, 0.1, N)
    E = 5000
    Te = 5
    # mi = 131.3 * constants.atomic_mass
    mi = 39.948 * constants.atomic_mass
    vb =  np.sqrt(constants.elementary_charge * 2.0 / mi)
    v0 = 2 * vb
    print(f"{v0}")
    n0 = 1.0e+09
    root = 1.0
    solution = np.zeros((2, N))


    for i in range(len(x_cell)):
        root = fixed_point_method(root, 1e-12, 10000, E, Te, vb, v0, x_cell[i])
        solution[0, i] = (n0 / root) / n0
        solution[1, i] = (v0 * root) / v0

    if plot_bool == True:
        plt.plot(x_cell, solution[0,:] , color = 'r', marker = "x", linestyle = "None", markevery = 15)
        plt.plot(x_cell, solution[1,:]  , 'b--')

        plt.yscale("log")
        plt.xlabel(fr"$x\;$[m]")
        plt.ylabel(fr"$ \log_{{10}} \frac{{n}}{{n_{0}}}; \log_{{10}} \frac{{v}}{{v_{0}}}$")

        plt.show()

    return solution


# sol = calc_analytical_solution(True, 1000)
