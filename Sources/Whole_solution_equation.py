import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
import os
from PlotCache import *

def get_solution(t, k, kg, M, beta, init_cond=[0,0,0,0]):
    from scipy.integrate import odeint
    
    def func(y, t, k, kg, M, beta):
        x1,x2,v1,v2 = y
        return [v1, v2, -(k+kg) * x1/M + kg * x2/M, kg * x1/M - (kg+k) * x2/M - beta * v2]
               #x1',x2',v1',                          v2'

    return odeint(func,init_cond,t,args=(k,kg,M,beta))

def PlotOne(ax,t,k,kg,M,beta):
    from types import ModuleType

    cache_name = f'Whole_{k}_{kg}_{M}_{beta}_{t[0]}_{t[-1]}.pkl'
    if IsInCache(cache_name):
        sol = ReadSolution(cache_name)
    else:
        sol = get_solution(t, k, kg, M, beta, [10,0,0,0])
        SaveSolution(sol, cache_name)
    ax.plot(t, sol[:,0], label=fr'x1(t), $\beta$={beta}')
    ax.plot(t, sol[:,1], label='x2(t)')
    
    #if we plot in a subplot, we must call set_xlabel
    #if we plot in a separate plot, we must call xlabel
    if isinstance(ax, ModuleType): #thanks Stackoverflow and Google:)
        ax.xlabel('t')
        ax.ylabel('x(t)')
    else:
        ax.set_xlabel('t')
        ax.set_ylabel('x(t)')
    ax.legend()
    #plt.savefig(os.path.join(os.getcwd(), 'Plots', f'Whole_solution_{k}_{kg}_{M}_{beta}.pdf'))

def grid_plot(k,kg,M,fname,t=None,betas=None,n_cols=3):
    plt.figure()
    #ranges
    if t is None:
        t = np.arange(1,30,0.1)
    if betas is None:
        betas = np.arange(1,10)

    fig, axes = plt.subplots(int(np.ceil(len(betas)/n_cols)), n_cols, figsize=(25,5))
    axes = np.ravel(axes)

    for ax, beta in zip(axes,betas):
        PlotOne(ax, t, k, kg, M, beta)
    plt.savefig(os.path.join(os.getcwd(), 'Plots', fname))
    #plt.show()

#fig. 3(a)
grid_plot(10, 5, 10,
          os.path.join(os.getcwd(), 'Plots', 'Fig_3_a.pdf'),
              t=np.arange(1,60,0.01), betas=[0,5,30])

#fig. 3(b)
fig, axes = plt.subplots(1, 3, figsize=(25,5))
PlotOne(axes[0], np.arange(1, 60, 0.01), 3, 30, 10, 0)
PlotOne(axes[1], np.arange(1, 60, 0.01), 30, 3, 10, 0)
PlotOne(axes[2], np.arange(1, 300, 0.01), 30, 3, 10, 0.05)
fig.savefig(os.path.join(os.getcwd(), 'Plots', 'Fig_3_b.pdf'))
plt.show()

