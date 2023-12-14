# coding: utf-8

# # Sample code for exercise 3-2
# 2017 Aug. Tsuyoshi Okubo  
# 2018 Dec. modified by TO
# 2020 Dec. modified by TO
#
# In this code, you can perform iTEBD simulation of the ground state of spin model on 1d chain.  
# $$\mathcal{H} = \sum_{i} J_z S_{i,z}S_{i+1,z} + J_{xy} (S_{i,x}S_{i+1,x} + S_{i,y}S_{i+1,y}) - h_x \sum_i S_{i,x} + D\sum_i S_{i,z}^2$$
# 
# Because we consider an infinitely long chain, the boundary condition is expected to be irrelevant.
#
# Note that, the accuracy of the calculation depends on chi_max, tau, and iteration steps.
# tau is gradually decreases from tau_max to tau_min
# 
# 
# You can change   
# 
# - (N: # of sites. In this case, our system is infinite)
# - m: size of spin  (2S = 2m + 1)  
# - Jz: amplitude of SzSz interaction  
# - Jxy: amplitude of SxSx + SySy interaction  
# - hx : amplitude of external field alogn x direction  
# - D : Single ion anisotropy  
# - (periodic: In this exercize, we only consider open boundary)
# - chi_max : maximum bond dimension of MPS
# - tau_max : maximum value of tau
# - tau_min : minimum value of tau
# - T_step : Total ITE steps
# - output_dyn_num : output data step


import numpy as np
import scipy.linalg as linalg
import TEBD
import iTEBD
from matplotlib import pyplot



m = 3         ## m = 2S + 1, e.g. m=3 for S=1 
Jz = 1.0      ## Jz for SzSz interaction
Jxy = 1.0     ## Jxy for SxSx + SySy interaction
hx = 0.0      ## external field along x direction
D = 0.0       ## single ion anisotropy

chi_max = 20  ## maxmum bond dimension at truncation

tau_max = 0.1     ## start imaginary time tau
tau_min = 0.001   ## final imaginary time tau
T_step=2000       ## ITE steps
output_dyn_num = 100 ## output steps



print("2S = m - 1, infinite spin chain")
print("m = "+repr(m))
print("Hamiltonian parameters:")
print("Jz = "+repr(Jz))
print("Jxy = "+repr(Jxy))
print("hx = "+repr(hx))
print("D = "+repr(D))

print("chi_max = "+repr(chi_max))

print("tau_max = "+repr(tau_max))
print("tau_min = "+repr(tau_min))
print("T_step = "+repr(T_step))
print("output_dyn_num = "+repr(output_dyn_num))



##iTEBD simulation
Tn, lam, T_list,E_list,mz_list = iTEBD.iTEBD_Simulation(m,Jz,Jxy,hx,D,chi_max,tau_max,tau_min,T_step,output_dyn=True,output_dyn_num=output_dyn_num)



## Calculate Energy
Env_left,Env_right = iTEBD.Calc_Environment_infinite(Tn,lam,canonical=False)
E_mps = iTEBD.Calc_Energy_infinite(Env_left,Env_right,Tn,lam,Jz,Jxy,hx,D)

print("iTEBD energy per bond = " + repr(E_mps))



## plot energy dynamics
pyplot.title("iTEBD Energy dynamics")
pyplot.plot(T_list[1:],E_list[1:],"o")
pyplot.xlabel("T")
pyplot.ylabel("E(T)")
pyplot.show()






