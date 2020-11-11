import numpy as np
import sympy as sp
from scipy import linalg
from matplotlib import pyplot as plt
import os
from PlotCache import *


t, k, k_g, M, beta = sp.symbols("t,k,k_g,M,beta")
g = (k_g)/(2*sp.sqrt(M*(k+k_g)))

A = sp.Matrix([[0, 1j*g], [1j*g, -beta/2]])

eigs = list(A.eigenvals().keys())

betas_range = np.linspace(0,10,200, dtype=complex)
M_ = k_ = kg_ = 1+0j

plt.rcParams.update({'font.size': 20})
plt.figure(figsize=(10,10))

cache_name = 'RWA_solution.pkl'
if IsInCache(cache_name):
    vals = ReadSolution(cache_name)
    
else:
    vals = [sp.lambdify(beta, eig.subs([(M,M_), (k,k_), (k_g, kg_)]))(betas_range)
            for eig in eigs]
    SaveSolution(vals, cache_name)
    
for i, v in enumerate(vals):
    plt.plot(betas_range, abs(np.real(v)), label=fr'$Re \lambda_{i+1}$', linewidth=3)
    plt.plot(betas_range, abs(np.imag(v)), label=fr'$Im \lambda_{i+1}$', linewidth=3)

plt.legend()
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\lambda_{1,2}$')
plt.savefig(os.path.join(os.getcwd(), 'Plots', 'Fig_4.pdf'))
plt.show()

