import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from odeintw import odeintw #may be not installed, install it with pip
from Lib.EasyPlot import ClassicAndRWAPlot
import os
from numba import jit

t = np.linspace(1, 400, 2000)
k=10; kg=1; beta=10; M=10
omega_d = np.sqrt((k+kg)/M)

#Get solution with RWA approximation
def get_solution(t, k, kg, M, F, beta, init_cond=[0,0,0,0]):
    omega = np.sqrt((k+kg)/M)
    
    #differential equations system
    @jit(nopython=True)
    def func(y, t, k, kg, M, beta, F, omega_d):
        a1, a2 = y
        force = 1j*np.sqrt(1/(2*M*omega))*(F/2)
        g = kg/(2*np.sqrt(M*(k+kg)))
        return [a1*1j*(omega_d-omega) + 1j*g*a2 - force, a2*1j*(omega_d-omega) + 1j*g*a1 - (beta/2)*a2]
        #return [(a2*1j*g) + force, (a1*1j*g) - (beta/2)*a2]
               #a1',                              a2'
    
    #parse initial conditions - convert x and v to a
    x1_0, x2_0, v1_0, v2_0 = init_cond
    p1_0, p2_0 = M*v1_0, M*v2_0
    a1_0 = x1_0 * np.sqrt(M*omega/2) + p1_0 / (1j*np.sqrt(2*M*omega))
    a2_0 = x2_0 * np.sqrt(M*omega/2) + p2_0 / (1j*np.sqrt(2*M*omega))
    y0=[a1_0, a2_0] 

    #get solution
    res = odeintw(func, y0, t, args=(k,kg,M,beta, F, omega_d))
    a1, a2 = res[:,0], res[:,1]
    for i, t in enumerate(t):
        a1[i] = a1[i] * np.exp(-1j*omega_d*t)
        a2[i] = a2[i] * np.exp(-1j*omega_d*t)
    a1_dag = np.conjugate(a1)
    a2_dag = np.conjugate(a2)
    x1 = np.sqrt(1/(2*M*omega)) * (a1 + a1_dag)
    x2 = np.sqrt(1/(2*M*omega)) * (a2 + a2_dag)
    return np.hstack((np.reshape(x1,(-1,1)), np.reshape(x2,(-1,1))))

#Classical solution
def get_solution_classic(t, k, kg, M, F, beta, init_cond=[0,0,0,0]): 
    @jit(nopython=True)
    def func(y, t, k, kg, M, beta, F, omega_d):
        x1, x2, p1, p2 = y
        return [p1/M, p2/M, -(k+kg)*x1 + kg*x2 - F*np.cos(omega_d * t), kg*x1 - (kg+k)*x2 - beta*p2]
               #x1',x2',          v1',                   v2'

    return odeint(func,init_cond,t,args=(k,kg,M,beta,F,omega_d))

p = ClassicAndRWAPlot('Simple_RWA')
p.set_xvalues(t, '$t$', '$x_{1,2}(t)$')
p.set_solver(get_solution)
p.set_solver_classic(get_solution_classic)
p.set_display_names(M='$M$', k='$k$', kg='$k_g$',
                    beta=r'$\beta$', omega_d=r'\omega_d', F='$F_0$')
p.figsize = (25,10)
p.plot_grid_wrapped(2, 'F', [0.01, 0.1, 1, 10, 100, 1000],
                    k=k, kg=kg, beta=beta, M=M)
plt.savefig(os.path.join(os.getcwd(), '..', 'images', 'Fig_7.pdf'))
if __name__ == '__main__': # do not show in run_all, where this file is imported
    plt.show()

