import numpy as np

def polynomial_degre(element_type):

    if element_type == "linear":
        PD = 1
        print(f"Using first order elements")

    if element_type == "quadratic":
        PD = 2
        print(f"Using second order elements")

    if element_type == "cubic":
        PD = 3
        print(f"Using third order elements")
        
    return PD


def S(xi, i, function_space ):
    
    if function_space == "linear":
        
        if i == 1:
            s = 0.5 * (1 - xi)
        elif i == 2:
            s = 0.5 * (1 + xi)

    elif function_space == "quadratic":

        if i == 1:
            s = 0.5 * (xi * (xi -1.0))
        elif i == 2:
            s = (1.0 - xi **2 )
        elif i == 3:
            s = 0.5 * (xi * (xi + 1))
            
    elif function_space == "cubic":
        if i == 1:
            s = ((-0.5625 * (xi ** 3)) + (0.5625 * (xi ** 2)) + (0.0625 * (xi)) -0.0625)
        if i == 2:
            s = ((1.6875 * (xi ** 3)) + (-0.5625 * (xi ** 2)) + (-1.6875 * (xi)) + 0.5625)
        if i == 3:
            s = ((-1.6875 * (xi ** 3)) + (-0.5625 * (xi ** 2)) + (1.6875 * (xi)) + 0.5625)
        if i == 4:
            s = ((0.5625 * (xi **3)) + (0.5625 * (xi ** 2)) + (-0.0625 * (xi)) - 0.0625)

    return s

def dS(xi, i, function_space):

    if function_space == "linear":

        if i == 1:
            s = -0.5 
        elif i == 2:
            s = 0.5 

    elif function_space == "quadratic":

        if i == 1:
            s = (xi - 0.5)
        elif i == 2:
            s = (- 2.0 * xi)
        elif i == 3:
            s = (0.5 + xi)

    elif function_space == "cubic":

        if i == 1:
            s = ((-1.6875 * (xi ** 2)) + (1.125 * xi) + 0.0625)
        if i == 2:
            s = ((5.0625 * (xi ** 2)) + (-1.125 * (xi)) - 1.6875)
        if i == 3:
            s = ((-5.0625 * (xi ** 2)) + (-1.125 * (xi)) + 1.6875 )
        if i == 4:
            s = ((1.6875 * (xi ** 2)) + (1.125 * (xi)) - 0.0625)

    return s

def U_h_plot(xi:float, i:int, P:list, function_space:str):

    if function_space == "linear":

        mat_2_solve = np.array([[-1.0, 1.0],
                                [1.0, 1.0]])

        if i == 1:
            RHS = np.array([P[0], 0.0])
            sol_linear = np.linalg.solve(mat_2_solve, RHS)
            s = sol_linear[0] * xi + sol_linear[1] 

        elif i == 2:
            RHS = np.array([0.0, P[1]])
            sol_linear = np.linalg.solve(mat_2_solve, RHS)
            s = sol_linear[0] * xi + sol_linear[1] 

    elif function_space == "quadratic":

        mat_2_solve = np.array([[1.0, -1.0, 1.0],
                                [0.0, 0.0, 1.0],
                                [1.0, 1.0, 1.0]])
        # print(f"Coef: {P}")

        if i == 1:
            RHS = np.array([P[0], 0.0, 0.0])
            sol_quadratic = np.linalg.solve(mat_2_solve, RHS)
            # print(sol_quadratic)
            s = (sol_quadratic[0] * (xi**2)) + (sol_quadratic[1]* xi) + sol_quadratic[2]

        elif i == 2:
            RHS = np.array([0.0, P[1], 0.0])
            sol_quadratic = np.linalg.solve(mat_2_solve, RHS)
            # print(sol_quadratic)

            s = (sol_quadratic[0] * (xi**2)) + (sol_quadratic[1] * xi) + sol_quadratic[2]

        elif i == 3:
            RHS = np.array([0.0, 0.0, P[2]])
            sol_quadratic = np.linalg.solve(mat_2_solve, RHS)
            # print(sol_quadratic)

            s = (sol_quadratic[0] * (xi**2)) + (sol_quadratic[1] * xi) + sol_quadratic[2]
            
    elif function_space == "cubic":
        mat_2_solve = np.array([[-1.0,  1.0, -1.0, 1.0],
                                [-1/27, 1/9, -1/3, 1.0], 
                                [ 1/27, 1/9,  1/3, 1.0], 
                                [ 1.0,  1.0,  1.0, 1.0]])

        if i == 1:
            RHS = np.array([P[0], 0.0, 0.0, 0.0])
            sol_cubic = np.linalg.solve(mat_2_solve, RHS)
            s = (sol_cubic[0] * (xi**3)) + (sol_cubic[1] * (xi**2)) + (sol_cubic[2] * xi) + sol_cubic[3]

        if i == 2:
            RHS = np.array([0.0, P[1], 0.0, 0.0])
            sol_cubic = np.linalg.solve(mat_2_solve, RHS)
            s = (sol_cubic[0] * (xi**3)) + (sol_cubic[1] * (xi**2)) + (sol_cubic[2] * xi) + sol_cubic[3]

        if i == 3:
            RHS = np.array([0.0, 0.0, P[2], 0.0])
            sol_cubic = np.linalg.solve(mat_2_solve, RHS)
            s = (sol_cubic[0] * (xi**3)) + (sol_cubic[1] * (xi**2)) + (sol_cubic[2] * xi) + sol_cubic[3]

        if i == 4:
            RHS = np.array([0.0, 0.0, 0.0, P[3]])
            sol_cubic = np.linalg.solve(mat_2_solve, RHS)
            s = (sol_cubic[0] * (xi**3)) + (sol_cubic[1] * (xi**2)) + (sol_cubic[2] * xi) + sol_cubic[3]

    return s

def U_h_int_lin(xi:float, P:list, jacobian:float):
        
        mat_2_solve = np.array([[-1.0, 1.0],
                                [1.0, 1.0]])

        RHS_1 = np.array([P[0], 0.0])
        RHS_2 = np.array([0.0, P[1]])

        l1_c = np.linalg.solve(mat_2_solve, RHS_1)
        l2_c = np.linalg.solve(mat_2_solve, RHS_2)

        s1 = (l1_c[0] * xi + l1_c[1])
        s2 = (l2_c[0] * xi + l2_c[1])

        return (s1 + s2) * jacobian


def U_h_int(xi:float, i:int, P:list, jacobian:float, function_space:str):

    if function_space == "linear":

        mat_2_solve = np.array([[-1.0, 1.0],
                                [1.0, 1.0]])

        if i == 1:
            RHS = np.array([P[0], 0.0])
            sol_linear = np.linalg.solve(mat_2_solve, RHS)
            s = (sol_linear[0] * xi + sol_linear[1]) * jacobian

        elif i == 2:
            RHS = np.array([0.0, P[1]])
            sol_linear = np.linalg.solve(mat_2_solve, RHS)
            s = (sol_linear[0] * xi + sol_linear[1] ) * jacobian

    elif function_space == "quadratic":

        mat_2_solve = np.array([[1.0, -1.0, 1.0],
                                [0.0, 0.0, 1.0],
                                [1.0, 1.0, 1.0]])

        if i == 1:
            RHS = np.array([P[0], 0.0, 0.0])
            sol_quadratic = np.linalg.solve(mat_2_solve, RHS)
            s = ((sol_quadratic[0] * xi**2) + (sol_quadratic[1]* xi) + sol_quadratic[2]) * jacobian

        elif i == 2:
            RHS = np.array([0.0, P[1], 0.0])
            sol_quadratic = np.linalg.solve(mat_2_solve, RHS)
            s = ((sol_quadratic[0] * xi**2) + (sol_quadratic[1] * xi) + sol_quadratic[2]) * jacobian

        elif i == 3:
            RHS = np.array([0.0, 0.0, P[2]])
            sol_quadratic = np.linalg.solve(mat_2_solve, RHS)
            s = ((sol_quadratic[0] * xi**2) + (sol_quadratic[1] * xi) + sol_quadratic[2]) * jacobian
            
    elif function_space == "cubic":
        mat_2_solve = np.array([[-1.0,  1.0, -1.0, 1.0],
                                [-1/27, 1/9, -1/3, 1.0], 
                                [ 1/27, 1/9,  1/3, 1.0], 
                                [ 1.0,  1.0,  1.0, 1.0]])
        if i == 1:
            RHS = np.array([P[0], 0.0, 0.0, 0.0])
            sol_cubic = np.linalg.solve(mat_2_solve, RHS)
            s = ((sol_cubic[0] * (xi**3)) + (sol_cubic[1] * (xi**2)) + (sol_cubic[2] * xi) + sol_cubic[3]) * jacobian

        if i == 2:
            RHS = np.array([0.0, P[1], 0.0, 0.0])
            sol_cubic = np.linalg.solve(mat_2_solve, RHS)
            s = ((sol_cubic[0] * (xi**3)) + (sol_cubic[1] * (xi**2)) + (sol_cubic[2] * xi) + sol_cubic[3]) * jacobian

        if i == 3:
            RHS = np.array([0.0, 0.0, P[2], 0.0])
            sol_cubic = np.linalg.solve(mat_2_solve, RHS)
            s = ((sol_cubic[0] * (xi**3)) + (sol_cubic[1] * (xi**2)) + (sol_cubic[2] * xi) + sol_cubic[3]) * jacobian

        if i == 4:
            RHS = np.array([0.0, 0.0, 0.0, P[3]])
            sol_cubic = np.linalg.solve(mat_2_solve, RHS)
            s = ((sol_cubic[0] * (xi**3)) + (sol_cubic[1] * (xi**2)) + (sol_cubic[2] * xi) + sol_cubic[3]) * jacobian

    return s

def x_2_xi(x, x_1, x_2, h_element):

    xi = (2.0 * x  - (x_1 + x_2)) / (h_element)

    return xi


def xi_to_x(xi, h_element, x_1, x_2):

    x = (h_element / 2.0) * xi + ((x_1 + x_2) / 2.0)

    return x