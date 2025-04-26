import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit

import basis_fun as BF

"""
FEM solver, solution to diggerential equation (DE) of the form:
        u*dT/dx -k *d^2T/dx^2 = F
        \Omega: 0 \leq x \geq 1
        BC: T(0) = 1; T(1) = 0
"""

def M_2_int(xsi, i, j, J, function_space):
    """
    Function arguments order:
        - i, j, J 
    """
    f = BF.S(xsi, i, function_space) * BF.S(xsi, j, function_space) * J

    return f

def K_2_int(xsi, i, j, J, k, function_space):
    """
    Function arguments order:
        - i, j, J, u, k 
    """
    f = ((k * BF.dS(xsi, i, function_space) * BF.dS(xsi, j, function_space) * ((1.0 / J)**2))) * J

    return f

def F_2_int(xsi, i, f, J, function_space):
    """
    Function arguments order:
        - i, j, J, f 
    """
    f = 0.0
    return f 

def assemble_M_K_mat(NE, NN, NBF, J_e, element_type, k):
        
    global_M_matrix = np.zeros((NN, NN))
    global_K_matrix = np.zeros((NN, NN))
    local_2_global_matrix = np.linspace(tuple(np.arange(0, NN)[0: NBF]), tuple(np.arange(0, NN)[-NBF::]), NE, dtype=int)


    for en in range(NE):
        # print(f"Assembling element matrix: {en + 1}")
        for i in range(NBF):
            for j in range(NBF):
                I_K = integrate.fixed_quad(K_2_int, -1.0, 1.0, args = (i + 1, j + 1, J_e, k, element_type), n=NBF)[0]
                I_M = integrate.fixed_quad(M_2_int, -1.0, 1.0, args = (i + 1, j + 1, J_e, element_type), n = NBF)[0]
                # local_K_matrix[i, j] = I
                # # print(f"K{i + 1}{j + 1}--> I: {I}")
                ki = local_2_global_matrix[en, i]
                kj = local_2_global_matrix[en, j]

                global_K_matrix[ki, kj] += I_K
                global_M_matrix[ki, kj] += I_M
    
    print(f"Stiffness and Mass matrices assembled!")

    return (global_M_matrix, global_K_matrix)


k = 0.7
## Variables for spatial domain
L = np.pi
NE = 128

## Variables for temporal domain
T_final = 2.0


BC_neumann_val = -5.0
BC_robin_val = 20.0
BC_robin_val_2 = 0.5

BC_type = "dirichlet"
# BC_type = "neumann"
# BC_type = "robin"

time_integration_types = ["forward_euler", "backward_euler", "crank_nicolson"]

dts = np.array([0.1, 0.05, 0.0025])

plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams.update({'font.size': 14})

element_type = "quadratic"
for time_integration_type in time_integration_types:
    rel_L2_arr = np.array([])

    for dt in dts:

        N_t = int(T_final / dt)

        PD = BF.polynomial_degre(element_type)
        NBF = PD + 1
        NN = PD * NE + 1

        global_BC_vector = np.zeros(NN)

        h_e = L / NE
        J_e = h_e / 2

        x_domain = np.linspace(0, L, NN)
        T = np.linspace(0, T_final, N_t)

        T_sol = np.zeros(NN)
        T_sol_i = np.sin(np.linspace(0.0, L, NN))
        T_sol_x_t = np.zeros((NN, N_t), dtype=np.float32)

        M_G_mat, K_G_mat = assemble_M_K_mat(NE, NN, NBF, J_e, element_type, k)

        T_sol_x_t[:, 0] = T_sol_i
        T_sol = T_sol_i.copy()
        if time_integration_type == "forward_euler":

            print(f"Solving with the {time_integration_type}")

            M_fe = M_G_mat[1:-1, 1:-1]
            K_fe = K_G_mat[1:-1, 1:-1]

            for t in range(1, N_t + 1): 
                # b = np.matmul((M_fe - dt * K_fe), T_sol[1:-1])
                a = dt * K_fe @ T_sol[1:-1]
                c = M_fe @ T_sol[1:-1]
                b = c - a
                T_sol_FE = np.linalg.solve(M_fe, b)

                T_sol_x_t[1:-1, t] = T_sol_FE
                T_sol[1:-1] = T_sol_FE

                T_sol[0] = 0.0
                T_sol[-1] = np.sin(np.pi)

        elif time_integration_type == "crank_nicolson":
            print(f"Solving with the {time_integration_type}")

            M = ((M_G_mat[1:-1, 1:-1]) + (0.5 * dt  * K_G_mat[1:-1, 1:-1]))
            LHS = (M_G_mat[1:-1, 1:-1] - 0.5 * dt  * K_G_mat[1:-1, 1:-1])

            for t in range(1, N_t):
                # print(f"Solving for time: {T[t]}")
                # b = LHS @ T_sol[1:-1]
                b = np.matmul(LHS, T_sol[1:-1])
                T_sol_FE = np.linalg.solve(M, b)
                T_sol_x_t[1:-1, t] = T_sol_FE
                T_sol[1:-1] = T_sol_FE

        elif time_integration_type == "backward_euler":
            print(f"Solving with the {time_integration_type}")

            M = M_G_mat[1:-1, 1:-1] + dt * K_G_mat[1:-1, 1:-1]
            for t in range(1, N_t):
                # print(f"Solving for time: {T[t]}")
                b = np.matmul(M_G_mat[1:-1, 1:-1], T_sol[1:-1]) 

                T_sol_FE = np.linalg.solve(M, b)
                T_sol_x_t[1:-1, t] = T_sol_FE
                T_sol[1:-1] = T_sol_FE







plt.rcParams['figure.figsize'] = [8, 5]
plt.rcParams.update({'font.size': 15})

##Plot solution
X, Y = np.meshgrid(T, x_domain)
Z = T_sol_x_t

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", linewidth=0, antialiased=False)
ax.set_ylabel(fr"$x\;$[m]")
ax.set_xlabel(fr"$t\;$[s]")
ax.set_zlabel(fr"$T\;$[K]")

ax.view_init(elev=32, azim=-49)
plt.tight_layout()
plt.show()




