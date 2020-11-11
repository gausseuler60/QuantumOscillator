import numpy as np
from matplotlib import pyplot as plt
from odeintw import odeintw #may be not installed, install it with pip
from PlotCache import *

#Get solution with RWA approximation
def get_solution(t, k, kg, M, beta, init_cond):
    cache_name = f'RWA_{k}_{kg}_{M}_{beta}_{t[0]}_{t[-1]}.pkl'
    if IsInCache(cache_name):
        return ReadSolution(cache_name)
        
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

    SaveSolution((x1,x2), cache_name)
    return (x1,x2)

#Classical solution
def get_solution_classic(t, k, kg, M, beta, init_cond):
    cache_name = f'RWA_{k}_{kg}_{M}_{beta}_{t[0]}_{t[-1]}_gauge.pkl'
    if IsInCache(cache_name):
        return ReadSolution(cache_name)
    
    from scipy.integrate import odeint
    
    def func(y,t,k,kg,M,beta):
        x1,x2,v1,v2 = y
        return [v1, v2, -(k+kg)*x1/M+kg*x2/M, kg*x1/M-(kg+k)*x2/M - beta*v2]
               #x1',x2',          v1',                   v2'
    sol = odeint(func,init_cond,t,args=(k,kg,M,beta))
    SaveSolution(sol, cache_name)
    return sol

def PlotOne(ax, x1_rwa, x2_rwa, x_classic, k, k_g, M, beta):
    ax.plot(t,x1_rwa, label='$x_1(t)$ RWA')
    ax.plot(t,x2_rwa, label='$x_1(t)$ RWA')
    ax.plot(t,x_classic[:,0],'--', label='$x_1(t)$ classic')
    ax.plot(t,x_classic[:,1],'--', label='$x_2(t)$ classic')
    
    ax.legend()
    ax.set_title(rf'$k={k}, k_g={k_g}, M={M}, \beta={beta}$')
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x(t)$')


#fig. 5
init_cond = [20,0,0,0]
t=np.linspace(0, 50, 150)

params = ([5, 0.1, 30, 1], [5, 10, 30, 1])

fig, ax = plt.subplots(1, 2, figsize=(25,10))
plt.rcParams.update({'font.size': 20})

for i, p in enumerate(params):
    x1, x2 = get_solution(t, *p, init_cond)
    classic = get_solution_classic(t, *p, init_cond)
    PlotOne(ax[i], x1, x2, classic, *p)

plt.savefig(os.path.join(os.getcwd(), 'Plots', 'Fig_5.pdf'))
plt.show()
