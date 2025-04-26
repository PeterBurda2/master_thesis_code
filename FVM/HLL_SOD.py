import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import os
from scipy import constants
import numba
import time

#spacial and time discretization
x = 1.0
numCells  = 1002
dx = x/numCells

#  CFL condition
cflScalar  = 0.1


# t = 1e-03
# timeSteps = 1000000
# dt = t/ timeSteps


########################################################
#            EQUATION TO SOLVE
#  SOD SHOCK TUBE EQN 
#   d(\rho, \rho*v, E)T/dt + d(\rho*v, \rho*v**2 + p, v*(E + p))T/dx = (0,0,0)T

#######################################################

# speed functions for HLL

@numba.njit
def lambdas_Speed(v:np.array, p:np.array, rho:np.array):
    gamma = 7/5
    c = np.sqrt((gamma * p) / rho)

    l1 = v - c
    l2 = v 
    l3 = v + c

    return l1, l3

@numba.njit
def PlusMinusSpeed(a):
    aplus = (a + np.abs(a))/2
    amunis = (a - np.abs(a))/2
    return aplus, amunis


@numba.njit
def aL_aR(vecVLambdasMin, vecVLambdasMax, i):
    minVals = np.zeros((1, 2), dtype= np.float64)
    maxVals = np.zeros((1, 2), dtype= np.float64)
#   min speed val
    minVals[0, 0] = vecVLambdasMin[i]
    minVals[0, 1] = vecVLambdasMin[i - 1]
    aL = minVals.min()
#   Max speed val 
    maxVals[0, 0] = vecVLambdasMax[i]
    maxVals[0, 1] = vecVLambdasMax[i - 1]
    aR = maxVals.max()

    return aL, aR


########## CONNTANTS ##########
print('Constants set up')
gamma = 7/5

############ linear decrease BC ################

print('Initial condition')

############ constant IC  ############
U0 = np.zeros((3, numCells))
# left side
U0[0, 0:int((numCells + 1) / 2)] = 1.0 #density
U0[1, 0:int((numCells + 1) / 2)] = 1.0 #pressure
U0[2, 0:int((numCells + 1) / 2)] = 0.0 #velocity

#right side 
U0[0, int((numCells + 1) / 2):numCells] = 0.125 #density
U0[1, int((numCells + 1) / 2):numCells] = 0.1 #pressure
U0[2, int((numCells + 1) / 2):numCells] = 0.0 #velocity

xscale = np.linspace(0,1, numCells)

plt.plot(xscale, U0[0,:], 'k', label = 'Density')
plt.plot(xscale, U0[1,:], 'r', label = 'Pressure')
plt.plot(xscale, U0[2,:], 'b', label = 'Speed')
plt.title('Initial conditions for Sod shock tube')
plt.legend()
plt.show()

@numba.njit
def Uvec_Fvec(U0, numCells):
    print('Arrays creation')

    Uvec = np.zeros((3, numCells))
    Fvec = np.zeros((3, numCells))
    gamma = 7/5

    rho = U0[0, :]
    pressure = U0[1, :]
    velocity = U0[2, :]

    epsilon =  (pressure)/( rho * (gamma - 1))
    E = U0[0, :] * ( epsilon + (0.5 * (velocity ** 2)))

    Uvec[0,:] = rho
    Uvec[1,:] = rho * velocity
    Uvec[2,:] = E

    Fvec[0,:] = rho * velocity
    Fvec[1,:] = (rho * velocity ** 2) + pressure
    Fvec[2,:] = velocity * (E + pressure)

    print('Arrays creation DONE')
    return Uvec, Fvec


def Fvec_Update(Uvec:np.array, Fvec:np.array):
    FvecN = np.zeros((3, len(Uvec[0])))
    gamma = 7/5
    rho = Uvec[0,:]
    speed = Uvec[1,:] / rho
    energy = Uvec[2,:]
    pressure = (Uvec[2,:] - (0.5 * (speed **2) * rho)) * (gamma - 1)

    FvecN[0, :] = speed * rho
    FvecN[1, :] = (speed ** 2) * rho + pressure
    FvecN[2, :] = speed * (energy + pressure)
    return FvecN


@numba.njit
def New_rho_p_v_return(Uvec):
    gamma = 7/5
    rho = Uvec[0,:]
    speed = Uvec[1,:] / rho
    pressure = (Uvec[2,:] - (0.5 * (speed **2) * rho)) * (gamma - 1)
    energy = Uvec[2,:]

    return rho, pressure, speed, energy

def U_F_BC(Uvec,Fvec):
    U_left = Uvec[:,0]
    U_right = Uvec[:,-1]

    F_left = Fvec[:,0]
    F_right = Uvec[:,-1]
    return U_left, U_right, F_left, F_right


Uvec, Fvec = Uvec_Fvec(U0, numCells)
FHLL = np.zeros((3, numCells - 1))
Unew = np.zeros((3, numCells))
tau = 0

@numba.njit
def NewVec(Uvec, FHLL, numCells):
    for ic in range(1, numCells - 1):
        Unew[:,ic] = Uvec[:,ic] - (dt/dx) * (FHLL[:,ic] - FHLL[:,ic - 1])
    return Unew

while tau < 0.150:

    r,p,v,e = New_rho_p_v_return(Uvec)
    l_min, l_max = lambdas_Speed(v,p,r)

    # print(l_max)

    amax = l_max.max()

    dt = (cflScalar * dx)/(amax)
    # print(f'dt = {dt}, a max = {amax}')
    
    for i in range(1, numCells):
        aL, aR = aL_aR(l_min, l_max, i)
        aplusR, aminusR = PlusMinusSpeed(aR)
        aplusL, aminusL = PlusMinusSpeed(aL)
        FHLL[:, i - 1] = ((aminusR - aminusL)/(aR - aL)) * Fvec[:, i] + ((aplusR - aplusL)/(aR - aL)) * Fvec[:, i - 1] - 0.5 * (((aR * np.abs(aL)) - (aL * np.abs(aR))) / (aR -aL)) * (Uvec[:, i] - Uvec[:, i-1])

    for ic in range(1, numCells - 1):
        Unew[:,ic] = Uvec[:,ic] - (dt/dx) * (FHLL[:,ic] - FHLL[:,ic - 1])

    Uvec[:, 1:(numCells-1)] = Unew[:,1:(numCells-1)].copy()

    tau = tau + dt
    print(tau)

    Fvec = Fvec_Update(Uvec, Fvec)



r,p,v,e = New_rho_p_v_return(Uvec)


fig, axs = plt.subplots(2, 2)
fig.set_size_inches(11, 6)

axs[0, 0].plot(xscale, r, 'r.')
axs[0, 0].set_title('Density')
axs[0, 0].set_xlabel(r'$x$ [m]')
axs[0, 0].set_ylabel(r'$\rho$ [$\mathrm{kg}/\mathrm{m}^3$]')

axs[0, 1].plot(xscale, p, 'k.')
axs[0, 1].set_title('Pressure')
axs[0, 1].set_xlabel(r'$x$ [m]')
axs[0, 1].set_ylabel(r'$P$ [Pa]')

axs[1, 0].plot(xscale, v, 'b.')
axs[1, 0].set_title('Velocity')
axs[1, 0].set_xlabel(r'$x$ [m]')
axs[1, 0].set_ylabel(r'$v$ [m/s]')

axs[1, 1].plot(xscale, e, 'g.')
axs[1, 1].set_title('Energy')
axs[1, 1].set_xlabel(r'$x$ [m]')
axs[1, 1].set_ylabel(r'$E$ [J]')

plt.tight_layout()
plt.show()
