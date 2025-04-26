import numpy as np
import matplotlib.pyplot as plt


def analytical_sol_dirichlet(x, u, k ,f, L):
    """
    Function arguments order:
        - x_vector, u, k, f, L,  
    """
    C2 = ((k * f * L) / u) / (1.0 - np.exp((u / k) * L))
    C1 = (k / u)**2 * f - C2

    T = ((k * f / u) * x) + C1 - ((k / u) ** 2 * f) + (C2 * np.exp(u / k * x))

    return T

def analytical_sol_neumann(x, u, k ,f, L, bc_val):
    """
    Function arguments order:
        - x_vector, u, k, f, L,  
    """

    C2 = (bc_val - (k * f / u)) * (k / u) * np.exp(-(u * L / k))
    C1 = ((k / u) ** 2 * f) - C2
    
    T = ((k * f / u) * x) + C1 - ((k / u) ** 2 * f) + (C2 * np.exp(u / k * x))

    return T

def analytical_sol_robin(x, u, k ,f, L, bc_val):

    """
    Function arguments order:
        - x_vector, u, k, f, L,  
    """

    C2 = (bc_val - (2.0 * k * f / u) - (k * f * L / u)) / (np.exp(u * L / k) + (2.0 * u * np.exp(u * L / k) / k) - 1.0)
    C1 = ((k / u) ** 2 * f) - C2

    T = ((k * f / u) * x) + C1 - ((k / u) ** 2 * f) + (C2 * np.exp(u / k * x))

    return T


def analytical_solution_time_dependent_x(x, t, k):
    u = np.sin(x) * np.exp(-k * t)

    return u

