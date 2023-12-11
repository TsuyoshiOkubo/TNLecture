# Sample code for MPS of random vector
# 2017 Augst Tsuyoshi Okubo
# 2018 Feb. Bug fixed 
# 2019 Jun. Modified the output
# 2020 Jun. Modified the output
# 2020 Oct. Modifed by TO
# 2020 Dec. Modifed by TO

import numpy as np
import scipy.linalg as linalg
import argparse
import copy
import MPS
from matplotlib import pyplot


#input from command line
def parse_args():
    parser = argparse.ArgumentParser(description='Test code for MPS apploximation for one dimensional spin model')
    parser.add_argument('-N', metavar='N',dest='N', type=int, default=10,
                        help='set system size N (default = 10)')
    parser.add_argument('-m', metavar='m',dest='m', type=int, default=2,
                        help='vector size m: total dimension is m^N (default = 2)')
    parser.add_argument('-chi', metavar='chi_max',dest='chi_max', type=int, default=20,
                        help='maximum bond dimension at truncation (default = 20)')
    parser.add_argument('-s', metavar='seed',dest='seed', type=int, default=None,
                        help='seed for random number generator (default = None)')
    return parser.parse_args()

def main():
    args = parse_args()
    N = args.N
    m = args.m
    chi_max = args.chi_max
    seed = args.seed
    if seed != None:
        np.random.seed(seed)
        
    

    print("Parameters: N, m, chi_max = "+repr(N)+", "+repr(m)+ ", "+repr(chi_max))
    print("Random seed: = "+repr(seed))
    eig_vec = ((np.random.rand(m**N)-0.5) + 1.0j * (np.random.rand(m**N)-0.5)).reshape(m**N)
    ## normalization
    norm = np.tensordot(eig_vec,eig_vec.conj(),axes=(0,0))
    eig_vec /= np.sqrt(np.abs(norm))

    ## Make exact MPS (from "left")
    Tn = []
    lam = [np.ones((1,))]
    lam_inv = 1.0/lam[0]
    R_mat = eig_vec[:].reshape(m,m**(N-1))

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

    ## Truncation 

    
    Tn_ex = copy.deepcopy(Tn)
    lam_ex = copy.deepcopy(lam)

    #Tn_ex = Tn
    #lam_ex = lam
    for i in range(N-1):
        chi = min(chi_max,lam[i+1].shape[0])
        lam[i+1]=lam[i+1][:chi]
        Tn[i]=Tn[i][:,:,:chi]
        Tn[i+1]=Tn[i+1][:,:chi,:]
        
    print("Truncation: chi_max = "+repr(chi_max))
    
    ## Distance between Exact MPS and truncated MPS

    #print("Distance between exact and truncated MPS = "+repr(np.sqrt(np.abs(1.0 + calc_innerproduct(Tn,lam,Tn,lam) - 2.0 * calc_innerproduct(Tn_ex,lam_ex,Tn,lam).real))))
    vec_ex = MPS.remake_vec(Tn_ex,lam_ex)
    vec_ap = MPS.remake_vec(Tn,lam)
    print("Distance between exact and truncated MPS = "+repr(linalg.norm(vec_ex - vec_ap)))

    ## plot Schmidt coefficient at N/2
    ## Red line indicates the position of chi_max
    pyplot.title("Schmidt coefficients for "+"(N, m) = ("+repr(N)+", "+repr(m)+") random vector")
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
