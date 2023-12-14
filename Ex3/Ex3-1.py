# coding: utf-8

# # Sample code for exercise 3-1
# 2017 Aug. Tsuyoshi Okubo  
# 2018 Dec. modified by TO
# 2020 Dec. modified by TO
#
# In this code, you can perform TEBD simulation of the ground state of spin model on 1d chain.  
# $$\mathcal{H} = \sum_{i} J_z S_{i,z}S_{i+1,z} + J_{xy} (S_{i,x}S_{i+1,x} + S_{i,y}S_{i+1,y}) - h_x \sum_i S_{i,x} + D\sum_i S_{i,z}^2$$
# 
# Note that, the accuracy of the calculation depends on chi_max, tau, and iteration steps.
# tau is gradually decreases from tau_max to tau_min
# 
# 
# You can change   
# 
# - N: # of sites
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
# - Perform_ED :flag to calculate exact ground state energy

import numpy as np
import scipy.linalg as linalg
import ED
import TEBD
from matplotlib import pyplot


N=10          ## Chain length 
m = 3         ## m = 2S + 1, e.g. m=3 for S=1 
Jz = 1.0      ## Jz for SzSz interaction
Jxy = 1.0     ## Jxy for SxSx + SySy interaction
hx = 0.0      ## external field along x direction
D = 0.0       ## single ion anisotropy
#periodic = False ## in this exersize , we only consider open boundary

chi_max = 20  ## maxmum bond dimension at truncation

tau_max = 0.1     ## start imaginary time tau
tau_min = 0.001   ## final imaginary time tau
T_step=2000       ## ITE steps
output_dyn_num = 100 ## output steps

## flag to calculate exact ground state enegy
## Note that for larger N, it is impossible to calculate exact energy
## In that casese, please set this flag False
Perform_ED = True


print("2S = m - 1, N-site spin chain")
print("N = "+repr(N))
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
print("Perform_ED = "+repr(Perform_ED))



## Obtain the smallest eigenvalue
if Perform_ED:
    eig_val,eig_vec = ED.Calc_GS(m,Jz, Jxy,hx,D,N,k=1)
    Eg = eig_val[0]/(N-1)



##TEBD simulation
Tn, lam,T_list,E_list,mz_list = TEBD.TEBD_Simulation(m,Jz,Jxy,hx,D,N,chi_max,tau_max,tau_min,T_step,output_dyn=True,output_dyn_num=output_dyn_num)



## Calculate Energy
Env_left,Env_right = TEBD.Calc_Environment(Tn,lam,canonical=False)
E_mps = TEBD.Calc_Energy(Env_left,Env_right,Tn,lam,Jz,Jxy,hx,D)

if Perform_ED:
    print("Ground state energy per bond = " +repr(Eg))
print("TEBD energy per bond = " + repr(E_mps))



## plot energy dynamics
pyplot.title("TEBD Energy dynamics")
pyplot.plot(T_list[1:],E_list[1:],"o")
pyplot.xlabel("T")
pyplot.ylabel("E(T)")
if Perform_ED:
    pyplot.axhline(y=Eg, color='red', label="Exact enegy")
    pyplot.legend()
pyplot.show()

