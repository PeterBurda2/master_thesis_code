import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.optimize import curve_fit

import basis_fun as BF
import analytical_fun as AF


"""
FEM solver, solution to diggerential equation (DE) of the form:
        u*dT/dx -k *d^2T/dx^2 = F
        \Omega: 0 \leq x \geq 1
        BC: T(0) = 1; T(1) = 0
"""

def K_2_int(xsi, i, j, J, u, k, function_space):
    """
    Function arguments order:
        - i, j, J, u, k 
    """
    f = ((BF.S(xsi, i, function_space) * u * BF.dS(xsi, j, function_space) * (1.0/ J)) + (k * BF.dS(xsi, i, function_space) * BF.dS(xsi, j, function_space) * ((1.0 / J)**2))) * J

    return f

def F_2_int(xsi, i, f, J, function_space):
    """
    Function arguments order:
        - i, j, J, f 
    """
    f = BF.S(xsi, i, function_space) * f * J
    return f 



## Variables for DE
u = 3.0
k = 1.0
f = 1.0
L = 1
NE = 10


BC_neumann_val = -5.0
BC_robin_val = 20.0
BC_robin_val_2 = 0.5

BC_type = "dirichlet"
# BC_type = "neumann"
# BC_type = "robin"


# element_types = ["linear", "quadratic", "cubic"]
element_types = ["linear"] #, "quadratic", "cubic"]


plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams.update({'font.size': 16})

for element_type in element_types:

    # element_type = "quadratic"
    PD = BF.polynomial_degre(element_type)

    # NEs = np.arange(a_i, b_f, 6, dtype=int)
    NBF = PD + 1
    NN = PD * NE + 1
    local_2_global_matrix = np.linspace(tuple(np.arange(0, NN)[0: NBF]), tuple(np.arange(0, NN)[-NBF::]), NE, dtype=int)
    local_K_matrix = np.zeros((NBF, NBF))
    local_F_vector = np.zeros((NBF, 1))

    global_K_matrix = np.zeros((NN, NN))
    global_F_vector = np.zeros((NN, 1))
    global_BC_vector = np.zeros((NN, 1))
    T_sol = np.zeros(NN)

    h_e = L / NE
    J_e = h_e / 2
    x_domain = np.linspace(0, L, NN)

    for en in range(NE):
        print(f"Assembling element matrix: {en + 1}")

        for i in range(NBF):
            for j in range(NBF):
                I = integrate.fixed_quad(K_2_int, -1.0, 1.0, args = (i + 1, j + 1, J_e, u, k, element_type), n=NBF)[0]
                
                # local_K_matrix[i, j] = I
                # # print(f"K{i + 1}{j + 1}--> I: {I}")
                ki = local_2_global_matrix[en, i]
                kj = local_2_global_matrix[en, j]

                global_K_matrix[ki, kj] += I
            
        
        for i in range(NBF):
            I_F = integrate.fixed_quad(F_2_int, -1.0, 1.0, args = (i + 1, f, J_e, element_type), n=NBF)[0]
            # local_F_vector[i,0] = I_F
            fi = local_2_global_matrix[en, i]

            global_F_vector[fi, 0] += I_F

    if BC_type == "dirichlet":
        # global_BC_vector[-1, 0] = -5.0

        T_sol[1:-1] = np.linalg.solve(global_K_matrix[1:-1, 1:-1], (global_F_vector[1:-1, 0] + global_BC_vector[1:-1, 0]))

        if element_type == "linear":

            plt.plot(np.linspace(0, L, 200), AF.analytical_sol_dirichlet(np.linspace(0, L, 200), u, k, f, L), color = "k", linewidth = 5, label = f"Analytical solution")

            plt.plot(x_domain, T_sol, color = "red", marker =  "s", markersize = 10, linestyle = "--" , linewidth = 2, label = f"FEM solution with the first order elements")
        
        if element_type == "quadratic":

            plt.plot(x_domain, T_sol, color = "lime", marker =  "o",  markersize = 9, linestyle = "-." ,linewidth = 1, label = f"FEM solution with the second order elements")
        
        if element_type == "cubic":

            plt.plot(x_domain, T_sol, color = "c", marker =  "^",  markersize = 8, linewidth = 1, label = f"FEM solution with the third order elements")




    elif BC_type == "neumann":

        global_BC_vector[-1, 0] = BC_neumann_val

        T_sol[1::] = np.linalg.solve(global_K_matrix[1::, 1::], (global_F_vector[1::, 0] + global_BC_vector[1::, 0]))

        if element_type == "linear":

            plt.plot(np.linspace(0, L, 200), AF.analytical_sol_neumann(np.linspace(0, L, 200), u, k, f, L, BC_neumann_val), color = "k", linewidth = 5, label = f"Analytical solution")

            plt.plot(x_domain, T_sol, color = "red", marker =  "s", markersize = 10, linestyle = "--" , linewidth = 2, label = f"FEM solution with the first order elements")
        
        if element_type == "quadratic":

            plt.plot(x_domain, T_sol, color = "lime", marker =  "o",  markersize = 9, linestyle = "-." ,linewidth = 1, label = f"FEM solution with the second order elements")
        
        if element_type == "cubic":

            plt.plot(x_domain, T_sol, color = "c", marker =  "^",  markersize = 8, linewidth = 1, label = f"FEM solution with the third order elements")




    elif BC_type == "robin":

        global_BC_vector[-1, 0] = BC_robin_val / 2
        global_K_matrix[-1, -1] += BC_robin_val_2

        T_sol[1::] = np.linalg.solve(global_K_matrix[1::, 1::], (global_F_vector[1::, 0] + global_BC_vector[1::, 0]))

        if element_type == "linear":

            plt.plot(np.linspace(0, L, 200), AF.analytical_sol_robin(np.linspace(0, L, 200), u, k, f, L, BC_robin_val), color = "k", linewidth = 5, label = f"Analytical solution")
            plt.plot(x_domain, T_sol, color = "red", marker =  "s", markersize = 10, linestyle = "--" , linewidth = 2, label = f"FEM solution with the first order elements")
        
        if element_type == "quadratic":

            plt.plot(x_domain, T_sol, color = "lime", marker =  "o",  markersize = 9, linestyle = "-." ,linewidth = 1, label = f"FEM solution with the second order elements")
        
        if element_type == "cubic":

            plt.plot(x_domain, T_sol, color = "c", marker =  "^",  markersize = 8, linewidth = 1, label = f"FEM solution with the third order elements")






plt.xlabel(fr"$x\;$[m]")
plt.ylabel(fr"$T\;$[K]")
plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=2)
plt.tight_layout()

# plt.show()



