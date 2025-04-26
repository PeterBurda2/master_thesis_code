import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

import cal_analytical_sol as cas



def calc_a_l_r(U, vb):

    lambda_left = np.ones((2, len(U[0, :]) - 1))
    lambda_right = np.ones((2, len(U[0, :]) - 1))
    

    lambda_left[0, :] = U[1, 0:-1] / U[0, 0:-1] - vb
    lambda_left[1, :] = U[1, 0:-1] / U[0, 0:-1] + vb

    lambda_right[0, :] = U[1, 1::] / U[0, 1::] - vb
    lambda_right[1, :] = U[1, 1::] / U[0, 1::] + vb


    lambda_left_min = np.minimum(lambda_left[0, :], lambda_left[1, :])
    lambda_right_min = np.minimum(lambda_right[0, :], lambda_right[1, :])

    lambda_left_max = np.maximum(lambda_left[0, :], lambda_left[1, :])
    lambda_right_max = np.maximum(lambda_right[0, :], lambda_right[1, :])

    a_Left = np.minimum(lambda_left_min, lambda_right_min)
    a_Right = np.maximum(lambda_left_max, lambda_right_max)

    return a_Left, a_Right

def a_l_pm(a_left):

    a_left_plus = (a_left + np.abs(a_left)) / 2
    a_left_minus = (a_left - np.abs(a_left)) / 2

    return a_left_minus, a_left_plus

def a_r_pm(a_right):

    a_right_plus = (a_right + np.abs(a_right)) / 2
    a_right_minus = (a_right - np.abs(a_right)) / 2

    return a_right_minus, a_right_plus

def calc_flux_interface(F_left, F_right, U_left, U_right, a_left_plus, a_left_minus, a_left, a_right_plus, a_right_minus, a_right):

    F_half = (((a_right_minus - a_left_minus) / (a_right - a_left)) * F_right) + (((a_right_plus - a_left_plus) / (a_right - a_left)) * F_left) - ( 0.5 * (((a_right * np.abs(a_left)) - (a_left * np.abs(a_right))) / (a_right - a_left)) * (U_right - U_left))
    # print(F_half)
    return F_half

def cacl_max_velocity(U, vb):

    v = U[1, :] / U[0, :]
    l_1 = v - vb
    l_2 = v + vb
    l_12_max = np.array([np.max(l_1), np.max(l_2)])

    a_max = np.max(l_12_max)

    return a_max


L = 0.1 #m
M = 1000 #DOMAIN number of cell
T = 1.0e-3 #s final time

CLF = 0.2
dx = L / M


E = 5000 #Vm^-1
T_e = 5 #eV
# M_i = 131.3 * constants.atomic_mass
M_i = 39.948 * constants.atomic_mass
q = constants.elementary_charge
vb = np.sqrt((q * T_e) / M_i)

#### IC ####
n0 = 1.0e+09
v0 = 2 * vb

F = np.ones((2, M + 2))

F[0, :] = F[0, :] * (n0 * v0)
F[1, :] = F[1, :] * ((n0 * (v0**2) )+ (n0 * (vb**2)))

U = np.ones((2, M + 2))
U[0, :] = U[0, :] * n0
U[1, :] = U[1, :] * (n0 * v0)

S = np.zeros((2, M + 2))
S[1, :] = ((n0 * q * E) / M_i)

t = 0
plt.rcParams.update({'font.size': 12})
plt.rcParams['figure.figsize'] = [9, 6]
# for i in range(0, 100):
while t < T:
    a_max = cacl_max_velocity(U, vb)
    dt = (CLF * dx) / a_max
    print(dt)

    F_L = F[:, 0:-1]
    F_R = F[:, 1:: ]
    U_L = U[:, 0:-1]
    U_R = U[:, 1:: ]


    a_left, a_right = calc_a_l_r(U, vb)
    a_l_m, a_l_p = a_l_pm(a_left)
    a_r_m, a_r_p = a_r_pm(a_right)

    F_h = calc_flux_interface(F_L, F_R, U_L, U_R, a_l_p, a_l_m, a_left, a_r_p, a_r_m, a_right)

    U[:, 1:-1] = U[:, 1:-1] - ((dt / dx) * np.diff(F_h)) + (dt * S[:, 1:-1])

    n = U[0, 1:-1]
    v = U[1, 1:-1] / U[0, 1:-1]
    

    ####UPDATE VECTORS####
    F[0, 1:-1] = (n * v)
    F[1, 1:-1] = ((n * (v**2)) + (n * (vb**2)))

    S[1, 1:-1] = (n * q * E) / M_i
    # print(S)
   
    #####BC#####
    # U[0, 0] = n0
    # U[1, 0] = n0 * v0
    
    U[0, -1] = U[0, -2]
    U[1, -1] = U[1, -2]

    # F[0, 0] = (n0 * v0)
    # F[1, 0] = ((n0 * (v0**2) )+ (n0 * vb))
    
    F[0, -1] = F[0, -2]
    F[1, -1] = F[1, -2]

    # S[1,0] = (n0 * q * E) / M_i
    S[1,-1] = S[1, -2]



    t += dt
    # print(i,F[:,0])

analytical_sol = cas.calc_analytical_solution(False, M)

plt.plot(np.linspace(0, L, M + 1), U[1, 0:-1] / (U[0, 0:-1] * v0), color = "green", linewidth = 2, label = fr"Numerical solution-HLL")
plt.plot(np.linspace(0, L, M ), analytical_sol[1, :], color = "green", marker = "x", markersize = 10, markevery = 15, label = fr"Analytical solution", linestyle = "None")


plt.plot(np.linspace(0, L, M + 1), U[0, 0:-1] / n0, color = "blue",  linewidth = 2, label = fr"Numerical solution-HLL")
plt.plot(np.linspace(0, L, M ), analytical_sol[0, :], color = "blue", marker = "x", markersize = 10, markevery = 15, label = fr"Analytical solution", linestyle = "None")

plt.xlabel(fr"$x\;$[m]", fontsize=15)
plt.ylabel(fr"$ \log_{{10}} \frac{{n}}{{n_{0}}}; \log_{{10}} \frac{{v}}{{v_{0}}}$", fontsize=15)

plt.legend()
plt.yscale("log")
plt.tight_layout()
plt.show()
