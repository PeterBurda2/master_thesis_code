import numpy as np
import matplotlib.pyplot as plt
import scipy.io

"""Solution to heat equation
    PDE: u_t = k * u_xx; k = 1.0
    BC: u(\pi, t) = u(0, t) = 0.0
    IC: u(x, 0) = sin(x)
"""
def u_analytical():
    x_corr = np.linspace(0.0, np.pi, 256)
    t_corr = np.linspace(0.0, 2.0, 100)
    u_sol = np.zeros((256, 100), dtype=np.float32 )
    # u_val = 0.0


    for j in range(len(t_corr)):
            # u_val = np.sin(x_corr) * np.exp(t_corr[j]) 
            u_sol[:, j] = np.sin(x_corr) * np.exp(-1.0 * t_corr[j]) 

    return u_sol

def test_sol():
    x_corr = np.linspace(0.0, np.pi, 256)
    t_corr = np.linspace(0.0, 1.0, 100)
    u_sol = np.zeros((256, 100), dtype=np.float32 )
    u_val = 0.0


    for j in range(len(t_corr)):
            # u_val = np.sin(x_corr) * np.exp(t_corr[j]) 
            u_sol[:, j] = np.sin(x_corr) * np.exp(-1.0 * t_corr[j]) 

    # return u_sol


    X, T =  np.meshgrid(t_corr, x_corr)

    ax = plt.axes(projection='3d')
    ax.plot_surface(X, T, u_sol, rstride=1, cstride=1,
                    cmap='coolwarm', edgecolor='none')
    plt.show()  

    data = {'x': x_corr,
            't': t_corr,
            'u_analytical': u_sol
            }
    scipy.io.savemat('HEQN_a_sol.mat', data)


