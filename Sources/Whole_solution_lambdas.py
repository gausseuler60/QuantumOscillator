import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
import Lib.EasyPlot as ep
import os

betas = np.linspace(0,10,200)

def CalculateSolution(betas, k_, kg_, M_):
  
    eigvals = np.zeros((len(betas),8))

    for r, beta_ in enumerate(betas):
        M = [[0,0,1,0], [0,0,0,1], [-(k_+kg_)/M_,kg_/M_,0,0], [kg_/M_,-(k_+kg_)/M_,0,-beta_]]
        v, _ = linalg.eig(M)
        for i, num in enumerate(sorted(v)):    
            eigvals[r, 2*i] = abs(np.real(num))
            eigvals[r, 2*i+1] = abs(np.imag(num))

    return eigvals

plt.figure()
lplt = ep.LambdasPlot('Simple_lambdas')
lplt.set_xvalues(betas, r'$\beta$', r'$\lambda_{1,2}$')
lplt.set_solver(CalculateSolution)
lplt.set_display_names(k_='$k$', kg_='$k_g$', M_='$M$')
lplt.plot_one(plt,kg_=1, k_=1, M_=1)

plt.savefig(os.path.join(os.getcwd(), '..', 'images', 'Fig_2.pdf'))
if __name__ == '__main__': # do not show in run_all, where this file is imported
    plt.show()

