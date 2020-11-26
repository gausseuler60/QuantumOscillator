import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
from . import PlotCache as pc
from types import ModuleType

class LambdasPlot:
    def __init__(self, cache_title):
        self.__cache_title = cache_title

    def __prepair_cache_name(self, arguments):   
        fname = self.__cache_title
        for par, val in arguments.items():
            fname+=f'_{val}'
        return fname+'.pkl'

    def set_beta_range(self, betas):
        self.__betas = betas

    def set_solver(self, solver_func):
        self.__CalculateSolution = solver_func

    def plot_one(self, ax, **arguments):
        fname = self.__prepair_cache_name(arguments)  
        if pc.IsInCache(fname):
            eigvals = pc.ReadSolution(fname)
        else:
            eigvals = self.__CalculateSolution(self.__betas, **arguments)
            pc.SaveSolution(eigvals, fname)
            
        num_eigs = eigvals.shape[1] // 2 # every one has 2 parts: real and image
        titles = np.hstack([(fr'Re $\lambda_{i+1}$', fr'Im $\lambda_{i+1}$') for i in range(num_eigs)])
        
        for i, tit in enumerate(titles):
            if isinstance(ax, ModuleType):
                ax.xlabel(r'$\beta$')
                ax.ylabel('Eigenvalues')
            else:
                ax.xlabel(r'$\beta$')
                ax.ylabel('Eigenvalues')
                print(eigvals[:,i])
            ax.plot(self.__betas, eigvals[:,i], linewidth=2, label=tit)
        ax.legend()

    def plot_grid(self, rows, swept_param, swept_values,  **arguments):
        params_dict = arguments

        fig, axes = plt.subplots(rows, len(swept_values) // rows, figsize=(25,10))
        axes = np.ravel(axes)
        for ax, par in zip(axes, swept_values):
            params_dict.update({swept_param: par})
            PlotOne(ax, x1, x2, classic, *params_dict)
        


'''def CalculateSolution(betas, M_, k_, kg_):
    
    eigvals = np.zeros((len(betas),8))

    for r, beta_ in enumerate(betas):
        M = [[0,0,1,0], [0,0,0,1], [-(k_+kg_)/M_,kg_/M_,0,0], [kg_/M_,-(k_+kg_)/M_,0,-beta_]]
        v, _ = linalg.eig(M)
        for i, num in enumerate(sorted(v)):    
            eigvals[r, 2*i] = abs(np.real(num))
            eigvals[r, 2*i+1] = abs(np.imag(num))

    return eigvals

betas_range = np.linspace(0, 10, 100)
lp = LambdasPlot('Whole_solution')
lp.set_beta_range(betas_range)
lp.set_solver(CalculateSolution, M_=1, k_=1, kg_=1)
plt.figure(figsize=(30,30))
lp.plot_one(plt)
plt.show()'''
