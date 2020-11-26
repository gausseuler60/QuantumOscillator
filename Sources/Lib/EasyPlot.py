import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt
from . import PlotCache as pc
from types import ModuleType

class CorePlot:
    def __init__(self, cache_title):
        self._display_names = {}
        self._cache_title = cache_title
        self._figsize = (25,10)

    def _prepair_cache_name(self, arguments):   
        fname = self._cache_title
        for par, val in arguments.items():
            fname+=f'_{val}'
        return fname+'.pkl'

    def _get_param_title(self, arg):
            return self._display_names.get(arg, arg)

    def _make_plot_title(self, arg_dict):
        return ', '.join([f'{self._get_param_title(a)}={b}' for a,b in arg_dict.items()])

    def _mark_axes(self, ax, arg_dict):
        if isinstance(ax, ModuleType):
            ax.xlabel(self._xtitle)
            ax.ylabel(self._ytitle)
            ax.title(self._make_plot_title(arg_dict))
        else:
            ax.set_xlabel(self._xtitle)
            ax.set_ylabel(self._ytitle)
            ax.set_title(self._make_plot_title(arg_dict))
                
    def set_xvalues(self, xvals, xtitle='x', ytitle='y'):
        self._xvals = xvals
        self._xtitle = xtitle
        self._ytitle = ytitle

    def set_solver(self, solver_func):
        self._CalculateSolution = solver_func

    @property
    def figsize(self):
        return self._figsize

    @figsize.setter
    def figsize(self, sz):
        self._figsize = sz

    def set_display_names(self, **names):
        self._display_names = names

    def plot_one(self, ax, **arguments):
        raise NotImplementedError('This is a base class, please override this method in a subclass.')

    def plot_grid_wrapped(self, rows, swept_param, swept_values,  **arguments):
        params_dict = arguments
    
        fig, axes = plt.subplots(rows, len(swept_values) // rows, figsize=self._figsize)
        axes = np.ravel(axes)
        for ax, par in zip(axes, swept_values):
            params_dict.update({swept_param: par})
            self.plot_one(ax, **params_dict)
            #ax.set_title(f'{self._get_param_title(swept_param)}={par}')

    def plot_grid_2d(self, rows,  **arguments):
        def _is_list(arg):
            return isinstance(arg, list) #\
                #or isinstance(arg, np.array) \
                    #or isinstance(arg, np.ndarray)
                    
        list_args = []
        list_arg_vals = []
        number_args = {}
        
        for arg, val in arguments.items():
            if _is_list(val):
                list_args.append(arg)
                list_arg_vals.append(val)
            else:
                number_args[arg] = val
                
        n_list_args = len(list_args)
        if n_list_args != 2:
            raise ValueError('One can plot only 2D-grid of parameters!')
        
        number_args = arguments
        par_name1, par_name2 = list_args
        par_value1, par_value2 = list_arg_vals
    
        fig, axes = plt.subplots(len(list_arg_vals[0]), len(list_arg_vals[1]), figsize=self._figsize)

        for i, val1 in enumerate(par_value1):
            for j, val2 in enumerate(par_value2):
                ax = axes[i,j]
                number_args.update({par_name1: val1, par_name2: val2})
                self.plot_one(ax, **number_args)
                ax.set_title(f'{self._get_param_title(par_name1)}={val1}, {self._get_param_title(par_name2)}={val2}')
        plt.tight_layout()

        
class LambdasPlot(CorePlot):
    def plot_one(self, ax, **arguments):
        fname = self._prepair_cache_name(arguments)  
        if pc.IsInCache(fname):
            eigvals = pc.ReadSolution(fname)
        else:
            eigvals = self._CalculateSolution(self._xvals, **arguments)
            pc.SaveSolution(eigvals, fname)
            
        num_eigs = eigvals.shape[1] // 2 # every one has 2 parts: real and image
        titles = np.hstack([(fr'Re $\lambda_{i+1}$', fr'Im $\lambda_{i+1}$') for i in range(num_eigs)])
        
        for i, tit in enumerate(titles):
            self._mark_axes(ax, arguments)
            ax.plot(self._xvals, eigvals[:,i], linewidth=2, label=tit)
        ax.legend()

class ClassicPlot(CorePlot):
    def plot_one(self, ax, **arguments):
        fname = self._prepair_cache_name(arguments)
        if pc.IsInCache(fname):
            sol = pc.ReadSolution(fname)
        else:
            sol = self._CalculateSolution(self._xvals, **arguments) 
        self._mark_axes(ax, arguments)
        ax.plot(self._xvals, sol[:,0], label=fr'x1(t), $\beta$={arguments["beta"]}')
        ax.plot(self._xvals, sol[:,1], label='x2(t)')
        ax.legend()

class ClassicAndRWAPlot(CorePlot):
    def _prepair_cache_name_classic(self, arguments):
        arg_new = arguments.copy()
        arg_new.update({1: 'gauge'}) #add tail to file name after args and before extension
        return super()._prepair_cache_name(arg_new)
    
    def set_solver_classic(self, solver_func):
        self._CalculateSolutionClassic = solver_func

    def plot_one(self, ax, **arguments):
        fname = self._prepair_cache_name(arguments)
        fname_classic = self._prepair_cache_name_classic(arguments) 
        if pc.IsInCache(fname):
            res_new = pc.ReadSolution(fname)
        else:
            res_new = self._CalculateSolution(self._xvals, **arguments)
            pc.SaveSolution(res_new, fname)
        if pc.IsInCache(fname_classic):
            res = pc.ReadSolution(fname_classic)
        else:
            res = self._CalculateSolutionClassic(self._xvals, **arguments)
            pc.SaveSolution(res, fname_classic)
        self._mark_axes(ax, arguments)
        ax.plot(self._xvals ,res_new[:,0], linewidth=2, label='$x_1(t)$ RWA')
        ax.plot(self._xvals, res_new[:,1], linewidth=2, label='$x_2(t)$ RWA')

        ax.plot(self._xvals, res[:,0],'--', label='$x_1(t)$ classic')
        ax.plot(self._xvals, res[:,1],'--', label='$x_2(t)$ classic')

        ax.legend()

        
