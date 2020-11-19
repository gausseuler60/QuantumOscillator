import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
from PlotCache import *


#A script for performong calculations in article <insert article name>
#Performs calculations, makes plots and saves them into PDF format.
#Calculated values are cached, so if you start a program again, data will be restored from cache
#Call this script with -u command-line option to recalculate and update cache

plt.rcParams.update({'font.size': 15})

betas = np.linspace(0,10,200)

def CalculateSolution():
    M_ = k_ = kg_ = 1

    
    eigvals = np.zeros((len(betas),8))

    for r, beta_ in enumerate(betas):
        M = [[0,0,1,0], [0,0,0,1], [-(k_+kg_)/M_,kg_/M_,0,0], [kg_/M_,-(k_+kg_)/M_,0,-beta_]]
        v, _ = linalg.eig(M)
        for i, num in enumerate(sorted(v)):    
            eigvals[r, 2*i] = abs(np.real(num))
            eigvals[r, 2*i+1] = abs(np.imag(num))

    return eigvals
        
fname = 'Whole_solution.pkl'
if IsInCache(fname):
    eigvals = ReadSolution(fname)
else:
    eigvals = CalculateSolution()
    SaveSolution(eigvals, fname)

titles = np.hstack([(fr'Re $\lambda_{i+1}$', fr'Im $\lambda_{i+1}$') for i in range(4)])
plt.figure(figsize=(10,10))
for i, tit in enumerate(titles):
    plt.xlabel(r'$\beta$')
    plt.ylabel('Eigenvalues')
    plt.plot(betas, eigvals[:,i], linewidth=2, label=tit)
plt.legend()  
plt.savefig(os.path.join(os.getcwd(), '..', 'images', 'Fig_2.pdf'))
plt.show()
