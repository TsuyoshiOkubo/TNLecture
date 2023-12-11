# Sample code for MPS of spin models
# 2017 Augst Tsuyoshi Okubo
# 2018 Feb. Bug fixed 
# 2019 Jan. Modified output
# 2020 Jan. Modified output
# 2020 Oct. Modified by TO
# 2020 Dec. Modified by TO

import numpy as np
import scipy.linalg as linalg
import ED
import MPS
import argparse
import copy
from matplotlib import pyplot


#input from command line
def parse_args():
    parser = argparse.ArgumentParser(description='Test code for MPS apploximation for one dimensional spin model')
    parser.add_argument('-N', metavar='N',dest='N', type=int, default=16,
                        help='set system size N (defalt = 16)')
    parser.add_argument('-Jz', metavar='Jz',dest='Jz', type=float, default=-1.0,
                        help='SzSz interaction (default = -1.0)')
    parser.add_argument('-Jxy', metavar='Jxy',dest='Jxy', type=float, default=0.0,
                        help='SxSx and SySy interactions (default = 0.0)')
    parser.add_argument('-m', metavar='m',dest='m', type=int, default=2,
                        help='Spin size m=2S +1 (default = 2)')
    parser.add_argument('-hx', metavar='hx',dest='hx', type=float, default=0.5,
                        help='extarnal magnetix field (default = 0.5)')
    parser.add_argument('-chi', metavar='chi_max',dest='chi_max', type=int, default=2,
                        help='maximum bond dimension at truncation (default = 2)')
    return parser.parse_args()


def main():
    args = parse_args()
    N = args.N
    m = args.m
    Jz = args.Jz
    Jxy = args.Jxy
    hx = args.hx
    chi_max = args.chi_max
    D = 0.0

    print("Model parameters: Jz, Jxy, hx = "+repr(Jz)+", "+repr(Jxy)+", "+repr(hx))
    print("Parameters: N, m, chi_max = "+repr(N)+", "+repr(m)+ ", "+repr(chi_max))

    
    eig_val,eig_vec = ED.Calc_GS(m,Jz,Jxy,hx,D,N,k=1)

    
    print("Ground state energy per bond = "+repr(eig_val[0]/(N-1)))


    ## Make exact MPS (from "left")
    Tn = []
    lam = [np.ones((1,))]
    lam_inv = 1.0/lam[0]
    R_mat = eig_vec[:,0].reshape(m,m**(N-1))

    chi_l=1
    for i in range(N-1):
        U,s,VT = linalg.svd(R_mat,full_matrices=False)
        chi_r = s.size

        Tn.append(np.tensordot(np.diag(lam_inv),U.reshape(chi_l,m,chi_r),(1,0)).transpose(1,0,2))
        lam.append(s)
        lam_inv = 1.0/s
        R_mat = np.dot(np.diag(s),VT).reshape(chi_r*m,m**(N-i-2))
        chi_l = chi_r
    Tn.append(VT.reshape(m,m,1).transpose(1,0,2))
    lam.append(np.ones((1,)))


    ## Calculate Energy
    Env_left=[]
    Env_right=[]
    for i in range(N):
        Env_left.append(np.identity((lam[i].shape[0])))
        Env_right.append(np.dot(np.dot(np.diag(lam[i+1]),np.identity((lam[i+1].shape[0]))),np.diag(lam[i+1])))
    
    Tn_ex = copy.deepcopy(Tn)
    lam_ex = copy.deepcopy(lam)
    E_exact = MPS.Calc_Energy(Env_left,Env_right,Tn,lam_ex,Jz,Jxy,hx,D)
    print("Energy of Exact MPS = "+repr(E_exact))

    #Tn_ex=Tn
    #lam_ex=lam
    ## Truncation 
    for i in range(N-1):
        chi = min(chi_max,lam[i+1].shape[0])
        lam[i+1]=lam[i+1][:chi]
        Tn[i]=Tn[i][:,:,:chi]
        Tn[i+1]=Tn[i+1][:,:chi,:]

    ## Calculate Energy
    Env_left=[]
    Env_right=[]
    for i in range(N):
        Env_left.append(np.identity((lam[i].shape[0])))
        Env_right.append(np.dot(np.dot(np.diag(lam[i+1]),np.identity((lam[i+1].shape[0]))),np.diag(lam[i+1])))

    print("Truncation: chi_max = "+repr(chi_max))
    E_truncated = MPS.Calc_Energy(Env_left,Env_right,Tn,lam,Jz,Jxy,hx,D)
    print("Energy of MPS with truncation = "+repr(E_truncated))
    print("Energy difference: E_truncated - E_exact =" + repr(E_truncated - E_exact))

    ## Distance between Exact MPS and truncated MPS
    #print("Distance between exact and truncated MPS = "+repr(np.sqrt(np.abs(1.0 + Report_Random.calc_innerproduct(Tn,lam,Tn,lam) - 2.0 * Report_Random.calc_innerproduct(Tn_ex,lam_ex,Tn,lam).real))))

    vec_ex = MPS.remake_vec(Tn_ex,lam_ex)
    vec_ap = MPS.remake_vec(Tn,lam)
    print("Distance between exact and truncated MPS = "+repr(linalg.norm(vec_ex - vec_ap)))
    

    
    ## plot Schmidt coefficient at N/2
    pyplot.title("Schmidt coefficients for "+repr(N)+" sites spin chain")
    pyplot.plot(np.arange(len(lam_ex[N//2]))+1,lam_ex[N//2]**2,"o",label="Schmidt coefficients")
    pyplot.axvline([chi_max],0,1,  c="red", linestyle='dashed', label="chi_max") ## position of chi_max
    pyplot.xlabel("index")
    pyplot.xscale("log")
    pyplot.ylabel("Schmidt coefficients")
    pyplot.yscale("log")
    pyplot.legend()
    pyplot.show()
    

    
if __name__ == "__main__":
    main()
