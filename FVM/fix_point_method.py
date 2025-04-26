import numpy as np
from scipy import constants
import matplotlib.pyplot as plt

def fixed_point_method_test():
    epsilon = 1e-12
    x_old = 2.0

    for i in range(0, 1000):

        x_new = 1.0/x_old + 1.0
        
        if np.abs(x_new - x_old) < epsilon:
            print(f"Root {x_new}")
            break

        x_old = x_new

# fixed_point_method_test()


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


N = 1001
x_cell = np.linspace(0.0, 0.1, N)
E = 300
Te = 2
mi = 131.3 * constants.atomic_mass
vb =  np.sqrt(constants.elementary_charge * 2.0 / mi)
v0 = 2 * vb
n0 = 1.0e+09
root = 1.0
solution = np.zeros((2, N))


for i in range(len(x_cell)):
    root = fixed_point_method(root, 1e-7, 10000, E, Te, vb, v0, x_cell[i])
    solution[0, i] = n0 / root
    solution[1, i] = v0 * root

plt.plot(x_cell, solution[0,:] / n0 , 'r--')
plt.plot(x_cell, solution[1,:] / v0 , 'b--')

plt.yscale("log")
plt.show()
