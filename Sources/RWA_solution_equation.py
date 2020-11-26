import numpy as np
from matplotlib import pyplot as plt
from odeintw import odeintw #may be not installed, install it with pip
from Lib.EasyPlot import ClassicAndRWAPlot
import os

x = np.linspace(1, 50, 1000)

#Get solution with RWA approximation
def get_solution(t, k, kg, M, beta, init_cond=[20,0,0,0]):
    omega = np.sqrt((k+kg)/M)
    
    #differential equations system
    def func(y, t, k, kg, M, beta):
        a1, a2 = y
        return [(a2*1j*kg)/(2*np.sqrt(M*(k+kg))), (a1*1j*kg)/(2*np.sqrt(M*(k+kg))) - (beta/2)*a2]
               #a1',                              a2'
    
    #parse initial conditions - convert x and v to a
    x1_0, x2_0, v1_0, v2_0 = init_cond
    p1_0, p2_0 = M*v1_0, M*v2_0
    a1_0 = x1_0 * np.sqrt(M*omega/2) + p1_0 / (1j*np.sqrt(2*M*omega))
    a2_0 = x2_0 * np.sqrt(M*omega/2) + p2_0 / (1j*np.sqrt(2*M*omega))
    y0=[a1_0, a2_0] 

    #get solution
    res = odeintw(func, y0, t, args=(k,kg,M,beta))
    a1, a2 = res[:,0], res[:,1]

    #Multiply by exponents (as was in ansatz)
    for i, t in enumerate(t):
        a1[i] = a1[i] * np.exp(-1j*omega*t)
        a2[i] = a2[i] * np.exp(-1j*omega*t)

    #Complex conjugated operators (annihilation operators)
    a1_dag = np.conjugate(a1)
    a2_dag = np.conjugate(a2)

    #Now restore x_1 and x_2
    x1 = np.sqrt(1/(2*M*omega)) * (a1 + a1_dag)
    x2 = np.sqrt(1/(2*M*omega)) * (a2 + a2_dag)

    return np.hstack((np.reshape(x1,(-1,1)), np.reshape(x2,(-1,1))))

#Classical solution
def get_solution_classic(t, k, kg, M, beta, init_cond=[20,0,0,0]): 
    from scipy.integrate import odeint
    
    def func(y,t,k,kg,M,beta):
        x1,x2,v1,v2 = y
        return [v1, v2, -(k+kg)*x1/M+kg*x2/M, kg*x1/M-(kg+k)*x2/M - beta*v2]
               #x1',x2',          v1',                   v2'
    sol = odeint(func,init_cond,t,args=(k,kg,M,beta))
    return sol

plt.figure()
p = ClassicAndRWAPlot('Simple_RWA')
p.set_xvalues(x, '$t$', '$x_{1,2}(t)$')
p.set_solver(get_solution)
p.set_solver_classic(get_solution_classic)
p.set_display_names(M='$M$', k='$k$', kg='$k_g$', beta=r'$\beta$')
p.figsize = (25,10)
p.plot_grid_wrapped(1, 'kg', [0.1, 10], k=5, M=30, beta = 1)
plt.savefig(os.path.join(os.getcwd(), '..', 'images', 'Fig_5.pdf'))
if __name__ == '__main__': # do not show in run_all, where this file is imported
    plt.show()

