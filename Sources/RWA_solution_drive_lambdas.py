import numpy as np
import sympy as sp
from matplotlib import pyplot as plt
import Lib.EasyPlot as ep
import os

betas = np.linspace(0, 20, 500, dtype=np.complex)


def CalculateSolution(betas, k_, kg_, M_, omega_d_):
    g, beta, omega_d, omega = sp.symbols("g, beta, omega_d, omega")

    g_ = (kg_)/(2*np.sqrt(M_*(k_+kg_)))
    omega_ = 1j * np.sqrt((k_+kg_)/M_)
    A = sp.Matrix([[1j*(omega_d-omega), 1j*g], [1j*g, 1j*(omega_d-omega)-beta/2]])

    eigs = list(A.eigenvals().keys())
    vals = [sp.lambdify(beta, eig.subs([(g,g_),(omega,omega_),(omega_d,omega_d_)]))(betas)
            for eig in eigs]

    eigvals = np.zeros((len(betas), len(vals) * 2))  # for real and complex parts

    for i, num in enumerate(vals):
        eigvals[:, 2 * i] = np.real(num)
        eigvals[:, 2 * i + 1] = np.imag(num)

    return eigvals


lplt = ep.LambdasPlot('RWA_solution_drive_lambdas')

lplt.set_xvalues(betas, r'$\beta$', r'$\lambda_{1,2}$')
lplt.set_solver(CalculateSolution)
lplt.set_display_names(k_='$k$', omega_d_=r'$\omega_d$')
lplt.plot_grid_2d(plt, M_=1, k_=[1, 5, 10], kg_=1, omega_d_=[0.01, 0.1, 5])

plt.savefig(os.path.join(os.getcwd(), '..', 'images', 'Fig_6.pdf'))
if __name__ == '__main__': # do not show in run_all, where this file is imported
    plt.show()
