import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
import Lib.EasyPlot as ep
import os

betas = np.linspace(0,10,200, dtype = np.complex)

def CalculateSolution(betas_range, k_, kg_, M_):
    t, k, k_g, M, beta = sp.symbols("t,k,k_g,M,beta")
    g = (k_g)/(2*sp.sqrt(M*(k+k_g)))

    A = sp.Matrix([[0, 1j*g], [1j*g, -beta/2]])

    eigs = list(A.eigenvals().keys())
    vals = [sp.lambdify(beta, eig.subs([(M,M_*(1+0j)), (k,k_*(1+0j)), (k_g, kg_*(1+0j))]))(betas_range)
            for eig in eigs]

    eigvals = np.zeros((len(betas), len(vals)*2)) #for real and complex parts
    
    for i, num in enumerate(vals):
        eigvals[:, 2*i] = np.abs(np.real(num))
        eigvals[:, 2*i+1] = np.abs(np.imag(num))
    return eigvals

plt.figure()
lplt = ep.LambdasPlot('Simple_RWA_motion')
lplt.set_xvalues(betas, r'$\beta$', r'$\lambda_{1,2}$')
lplt.set_display_names(k_='$k$', kg_='$k_g$', M_='$M$')
lplt.set_solver(CalculateSolution)
lplt.plot_one(plt, k_=1, kg_=1, M_=1)
plt.savefig(os.path.join(os.getcwd(), '..', 'images', 'Fig_4.pdf'))
if __name__ == '__main__': # do not show in run_all, where this file is imported
    plt.show()


