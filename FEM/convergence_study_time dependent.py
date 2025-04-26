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

def U_h_int_lin(xi: float, P: np.array):
    """Linear interpolation using mapping to [-1,1]"""
    nodes = np.array([-1, 1])  # Nodes at endpoints
    V = np.vander(nodes, increasing=True)
    coeffs = np.linalg.solve(V, P)
    return coeffs[0] + coeffs[1]*xi

def U_h_int_quad(xi: float, P: np.array):
    """Quadratic interpolation using mapping to [-1,1]"""
    nodes = np.array([-1, 0, 1])  # Nodes at endpoints and center
    V = np.vander(nodes, increasing=True)
    coeffs = np.linalg.solve(V, P)
    return coeffs[0] + coeffs[1]*xi + coeffs[2]*xi**2

def U_h_int_cub(xi: float, P: np.array):
    """Cubic interpolation using mapping to [-1,1]"""
    nodes = np.array([-1, -1/3, 1/3, 1])
    V = np.vander(nodes, increasing=True)
    coeffs = np.linalg.solve(V, P)
    return coeffs[0] + coeffs[1]*xi + coeffs[2]*xi**2 + coeffs[3]*xi**3
    

def M_2_int(xsi, i, j, J, function_space):
    """
    Function arguments order:
        - i, j, J 
    """
    IM = BF.S(xsi, i, function_space) * BF.S(xsi, j, function_space) * J

    return IM
def K_2_int(xsi, i, j, J, k, function_space):
    """
    Function arguments order:
        - i, j, J, u, k 
    """
    IK = ((k * BF.dS(xsi, i, function_space) * BF.dS(xsi, j, function_space) * ((1.0 / J) ** 2))) * J

    return IK


def assemble_M_K_mat(NE, NN, NBF, J_e, element_type, k):
        
    global_M_matrix = np.zeros((NN, NN))
    global_K_matrix = np.zeros((NN, NN))
    local_2_global_matrix = np.linspace(tuple(np.arange(0, NN)[0: NBF]), tuple(np.arange(0, NN)[-NBF::]), NE, dtype=int)

    print(f"element type: {element_type}")
    for en in range(NE):
        # print(f"Assembling element matrix: {en + 1}")
        for i in range(NBF):
            for j in range(NBF):
                I_K = integrate.fixed_quad(K_2_int, -1.0, 1.0, args = (i + 1, j + 1, J_e, k, element_type), n=NBF)[0]
                I_M = integrate.fixed_quad(M_2_int, -1.0, 1.0, args = (i + 1, j + 1, J_e, element_type), n = NBF + 1)[0]
                # local_K_matrix[i, j] = I
                # # print(f"K{i + 1}{j + 1}--> I: {I}")
                ki = local_2_global_matrix[en, i]
                kj = local_2_global_matrix[en, j]

                global_K_matrix[ki, kj] += I_K
                global_M_matrix[ki, kj] += I_M
    
    # print(f"Stiffness and Mass matrices assembled!")

    return (global_M_matrix, global_K_matrix)


def analytical_sol_t(xi, k, t, h_element, x_1, x_2):
    """
    Function arguments order:
        - x_vector, u, k, f, L,  
    """
    # print(f"{k}, {t}")
    x = (h_element / 2.0) * xi + ((x_1 + x_2) / 2.0)
    Temp = np.sin(x) * np.exp(-k * t)

    return Temp
def analytical_sol_t_2(x, t, k):
    """
    Function arguments order:
        - x_vector, u, k, f, L,  
    """
    T = np.sin(x) * np.exp(-k * t)

    return T
def integrant(xi, k, t, h_element, x_1, x_2, P, jacobian, element_type):
    # print(f"{P}")
    
    if element_type == "linear":

        Int = (analytical_sol_t(xi, k, t, h_element, x_1, x_2) - U_h_int_lin(xi, P)) ** 2 * jacobian

    elif element_type == "quadratic":

        Int = (analytical_sol_t(xi, k, t, h_element, x_1, x_2) - U_h_int_quad(xi, P)) ** 2 * jacobian
    
    elif element_type == "cubic":

        Int = (analytical_sol_t(xi, k, t, h_element, x_1, x_2) - U_h_int_cub(xi, P)) ** 2 * jacobian


    return Int



## Variables for PDE
# u = 3.0
k = 0.7
# f = 1.0

## Variables for spatial domain
L = np.pi

element_type = "cubic"

## Variables for temporal domain
T_final = 2.0
plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams.update({'font.size': 14})

time_integration_types = ["forward_euler", "backward_euler", "crank_nicolson"]

colors = ["r", "g", "b"]
ci = 0
tau = 0.001
dts = np.array([tau, tau / 2, tau / 4, tau / 8, tau / 16 ])

# NE = 128


for time_integration_type in time_integration_types:
    rel_L2_arr = np.array([])

    if time_integration_type == "forward_euler":
        NE = 6
    else:
        NE= 200
    
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
        T_sol_i = np.sin(x_domain)
        T_sol_x_t = np.zeros((NN, N_t+1), dtype=np.float32)
        T_sol_x_t[1:-1, 0] = T_sol_i[1:-1]
        T_sol = T_sol_i.copy()

        M_G_mat, K_G_mat = assemble_M_K_mat(NE, NN, NBF, J_e, element_type, k)


        print(f"dt {dt}")

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

            c = 0
            I_error_global = 0.0
            # print(T_final)
            for en in range(NE):
                # print()
                I_err_element = integrate.fixed_quad(
                    integrant, -1.0, 1.0,
                    args=(k, T_final, h_e, x_domain[c], x_domain[c + PD], T_sol[c: c + PD + 1], J_e, element_type),
                    n=NBF
                )[0]
                I_error_global += I_err_element 

            L2_error = np.sqrt(I_error_global)
            T_exact = analytical_sol_t_2(x_domain, T_final, k)

            # L2_error_numeric = np.sqrt(np.sum((T_sol - T_exact)**2) * (h_e / 2))
            # I_g = np.sqrt((T_exact - T_sol) @ M_fe @ (T_exact - T_sol)) 

            print(f"Error: {L2_error}; for dt: {dt}")

            rel_L2_arr = np.append(rel_L2_arr, L2_error)
            plot_name = "Forward Euler" 

        elif time_integration_type == "crank_nicolson":
            print(f"Solving with the {time_integration_type}")

            # System matrices for interior nodes only
            M_int = M_G_mat[1:-1, 1:-1]
            K_int = K_G_mat[1:-1, 1:-1]

            # Pre-compute matrices
            A = M_int + (0.5 * dt * K_int)
            B = M_int - (0.5 * dt * K_int)

            for t in range(1,N_t+1):
                # print("HERE")
                b = np.matmul(B, T_sol[1:-1])
                T_sol_CN = np.linalg.solve(A, b)  # Solve linear system
                T_sol_x_t[1:-1, t] = T_sol_CN
                T_sol[1:-1] = T_sol_CN

                # Apply boundary conditions explicitly
                T_sol[0] = 0.0
                T_sol[-1] = np.sin(np.pi)
                # T_sol[-1] = 0.0

            # Compute exact analytical solution at final time
            T_exact = analytical_sol_t_2(x_domain, T_final, k)

            # Compute global L2 norm error accurately
            c = 0
            I_error_global = 0.0
            print(T_final)
            for en in range(NE):
                # print()
                I_err_element = integrate.fixed_quad(
                    integrant, -1.0, 1.0,
                    args=(k, T_final, h_e, x_domain[c], x_domain[c + PD], T_sol[c: c + PD + 1], J_e, element_type),
                    n=NBF
                )[0]
                I_error_global += I_err_element  


            L2_error = np.sqrt(I_error_global)
            # L2_error_numeric = np.sqrt(np.sum((T_sol - T_exact)**2) * (h_e / 2))
            # I_g = np.sqrt((T_exact - T_sol) @ M_G_mat @ (T_exact - T_sol)) 

            print(f"Error: {L2_error}; for dt: {dt}")

            rel_L2_arr = np.append(rel_L2_arr, L2_error)
            plot_name = "Crank-Nicolson" 
            

        elif time_integration_type == "backward_euler":
            print(f"Solving with the {time_integration_type}")

            M = M_G_mat[1:-1, 1:-1] + dt * K_G_mat[1:-1, 1:-1]
            for t in range(1, N_t + 1):
                # print(f"Solving for time: {T[t]}")
                b = np.matmul(M_G_mat[1:-1, 1:-1], T_sol[1:-1]) 

                T_sol_FE = np.linalg.solve(M, b)
                T_sol_x_t[1:-1, t] = T_sol_FE
                T_sol[1:-1] = T_sol_FE

                T_sol[0] = 0.0
                T_sol[-1] = np.sin(np.pi)

            c = 0
            I_error_global = 0
        
            T_exact = analytical_sol_t_2(x_domain, T_final, k)

            for en in range(NE):
                I_err = integrate.fixed_quad(integrant, -1.0, 1.0, args = (k, T_final, h_e, x_domain[c], x_domain[c + PD], T_sol[c: c + (PD + 1)], J_e, element_type), n=NBF)[0]
                I_error_global += I_err
                c += PD
            
            I_g = np.sqrt((T_exact - T_sol) @ M_G_mat @ (T_exact - T_sol))
            rel_L2_arr = np.append(rel_L2_arr, np.sqrt(I_error_global))

            print(f"Error: {np.sqrt(I_error_global)}; for dt: {dt}")
            plot_name = "Backward Euler" 

        

    slope, intercept = np.polyfit(np.log(dts), np.log(rel_L2_arr), 1)

    
    if time_integration_type == "crank_nicolson":
        plt.plot(dts, rel_L2_arr, color = colors[ci], marker = "s", linestyle = "-", label = f"Method: {plot_name}, slope k = {np.round(slope, 1)}")
        plt.plot(dts, dts**2 * np.exp(intercept + 0.8), color = colors[ci], linestyle = "--", label = f"Reference slope k = {2.0}")
    elif time_integration_type == "backward_euler":
        plt.plot(dts, dts**1 * np.exp(intercept + 0.8), color = colors[ci], linestyle = "--", label = f"Reference slope k = {1.0}")
        plt.plot(dts, rel_L2_arr, color = colors[ci], marker = "s", markersize = 8, linestyle = "-", label = f"Method: {plot_name}, slope k = {np.round(slope, 1)}")
    elif time_integration_type == "forward_euler":
        plt.plot(dts, dts**1 * np.exp(intercept + 0.8), color = colors[ci], linestyle = "--", label = f"Reference slope k = {1.0}")
    ci+=1

    print(slope)
    plt.xlabel(fr"$\Delta t\;$[s]")
    plt.ylabel(r"$E_{2}(||u_{h}  - u_{\mathrm{A}}||^{2})$")
    plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
                    mode="expand", borderaxespad=0, ncol=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(which="both", color = "0.9")
    plt.tight_layout()


plt.show()






