import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
from Lib.EasyPlot import ClassicPlot
import os

t=np.arange(1,60,0.01)

def get_solution(t, k, kg, M, beta, init_cond=[10,0,0,0]):
    from scipy.integrate import odeint
    
    def func(y, t, k, kg, M, beta):
        x1,x2,v1,v2 = y
        return [v1, v2, -(k+kg) * x1/M + kg * x2/M, kg * x1/M - (kg+k) * x2/M - beta * v2]
               #x1',x2',v1',                          v2'

    return odeint(func,init_cond,t,args=(k,kg,M,beta))


plt.figure()
#fig. 3(a)
p = ClassicPlot('Simple')
p.set_xvalues(t, '$t$', '$x_{1,2}(t)$')
p.set_solver(get_solution)
p.set_display_names(M='$M$', k='$k$', kg='$k_g$', beta=r'$\beta$')
p.figsize = (25,5)
p.plot_grid_wrapped(1, 'beta',[0,5,30], k=10, kg=5, M=10)
plt.savefig(os.path.join(os.getcwd(), '..', 'images', 'Fig_3_a.pdf'))
if __name__ == '__main__': # do not show in run_all, where this file is imported
    plt.show()

plt.figure()
#fig. 3(b)
fig, axes = plt.subplots(1, 3, figsize=(25,5))
p.plot_one(axes[0], k=3, kg=30, M=10, beta=0)
p.plot_one(axes[1], k=30, kg=3, M=10, beta=0)
p.plot_one(axes[2], k=30, kg=3, M=10, beta=0.05)
fig.savefig(os.path.join(os.getcwd(), '..', 'images', 'Fig_3_b.pdf'))
if __name__ == '__main__': # do not show in run_all, where this file is imported
    plt.show()


