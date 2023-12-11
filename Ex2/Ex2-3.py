# Sample code for MPS of a vector representing an image
# 2020 Oct. Tsuyoshi Okubo
# 2020 Dec. Modified by TO

import numpy as np
import scipy.linalg as linalg
import argparse
import copy
import matplotlib.pyplot as plt
from PIL import Image ## Python Imaging Library
import MPS

#input from command line
def parse_args():
    parser = argparse.ArgumentParser(description='Sample code for MPS apploximation for a vector representing an image. The original $(X, Y)$ pixel image is firstly transformed into the gray scale. Then, from it we cut $(m^{N/2}, m^{N/2})$ region for even $N$. In the case of odd $N$, we cut $(m^{(N+1)/2}, m^{(N-1)/2})$. We can consider the trimmed image as am $m^N$ dimensional vector, which is used as the input vector in the following analysis.')
    parser.add_argument('-N', metavar='N',dest='N', type=int, default=16,
                        help='set system size N (default = 16)')
    parser.add_argument('-m', metavar='m',dest='m', type=int, default=2,
                        help='vector size m: total dimension is m^N (default = 2)')
    parser.add_argument('-chi', metavar='chi_max',dest='chi_max', type=int, default=20,
                        help='maximum bond dimension at truncation (default = 20)')
    parser.add_argument('-f', '--file',metavar='filename',dest='filename', default="sample.jpg",
                        help='filename of the image. (default: sample.jpg)')
    return parser.parse_args()


def main():
    args = parse_args()
    N = args.N
    m = args.m
    chi_max = args.chi_max
    filename = args.filename
    

    img = Image.open(filename) ## load image
    img_gray = img.convert("L") ## convert to grayscale

    img_x, img_y = img_gray.size
    if N % 2 == 0:
        new_x = m**(N//2)
        new_y = m**(N//2)
    else:
        new_x = m**((N+1)//2)
        new_y = m**((N-1)//2)

    ## trimming
    img_gray_trimmed = img_gray.crop(((img_x - new_x)//2, (img_y - new_y)//2, (img_x + new_x)//2, (img_y + new_y)//2))

    img_gray_trimmed.save("./gray_trimmed.png") ## save grayscale image


    print("Parameters: N, m, chi_max = "+repr(N)+", "+repr(m)+ ", "+repr(chi_max))
    print("Input file: " + filename) ## print array shape
    print("Original size: "+repr(img_gray.size)) ## print array shape
    print("Trimmed size: " +repr(img_gray_trimmed.size))


    # plt.axis("off")
    # plt.title("Original input image")
    # plt.imshow(img_gray,cmap='gray')
    # plt.show()

    # plt.axis("off")
    # plt.title("Trimmed input image")
    # plt.imshow(img_gray_trimmed,cmap='gray')
    # plt.show()

    ## make vector
    img_vec = np.array(img_gray_trimmed,dtype=float).flatten()

    ## normalization
    img_norm = np.linalg.norm(img_vec)
    img_vec /= img_norm


    ## Make exact MPS (from "left")
    Tn = []
    lam = [np.ones((1,))]
    lam_inv = 1.0/lam[0]
    R_mat = img_vec[:].reshape(m,m**(N-1))

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


    ## plot Schmidt coefficient at N/2
    ## Red line indicates the position of chi_max
    plt.title("Schmidt coefficients for "+"(N, m) = ("+repr(N)+", "+repr(m)+") picture image")
    plt.plot(np.arange(len(lam_ex[N//2]))+1,lam_ex[N//2]**2,"o",label="Schmidt coefficients")
    plt.axvline([chi_max],0,1,  c="red", linestyle='dashed', label="chi_max") ## position of chi_max
    plt.xlabel("index")
    plt.xscale("log")
    plt.ylabel("Schmidt coefficients")
    plt.yscale("log")
    plt.legend()
    #plt.show()

    vec_ex = MPS.remake_vec(Tn_ex,lam_ex)
    vec_ap = MPS.remake_vec(Tn,lam)
    print("Distance between exact and truncated MPS = "+repr(linalg.norm(vec_ex - vec_ap)))

    ## Original and Approximated image
    img_ex = Image.fromarray(np.uint8(np.clip(img_norm*vec_ex.reshape((new_y,new_x)),0,255)))
    img_ap = Image.fromarray(np.uint8(np.clip(img_norm*vec_ap.reshape((new_y,new_x)),0,255)))

    img_ap.save("./gray_trimmed_ap.png") ## save grayscale image

    plt.figure(figsize=(2*new_x*0.01,new_y*0.01))
    plt.subplot(1,2,1)
    plt.axis("off")
    plt.title("original")
    plt.imshow(img_ex,cmap='gray')

    plt.subplot(1,2,2)
    plt.axis("off")
    plt.title("approximated")
    plt.imshow(img_ap,cmap='gray')
    plt.show()    
if __name__ == "__main__":
    main()
