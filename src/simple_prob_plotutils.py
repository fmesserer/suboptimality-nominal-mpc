import os
from typing import Optional, Union, List, Tuple
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import casadi as ca
from latexify import latexify
from solver_treeMPC import treeOCP, ScenarioTraj
from utils_save_load import ResultsValFunc


def set_plot_defaults(outfolder:str='plots/') -> str:

    colors = sns.color_palette('colorblind')
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colors) 
    latexify(fig_width=3.5)
    # create folder if it does not exist
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)
    return outfolder


def eval_stage_state_cost(prob: treeOCP, x_vals: np.ndarray) -> np.ndarray:
    
    lk_func = ca.Function("lk_func", [prob.x, prob.u, prob.p], [prob.lk_expr])
    return lk_func(x_vals, 0, 0).full().flatten()


def plot_stage_state_cost(prob: treeOCP, saveas=None, show=False) -> None:

    x_lb = -.2
    x_ub = .5
    x_ls = np.linspace(x_lb, x_ub, 300)
    lk_ls = eval_stage_state_cost(prob, x_ls)

    plt.figure()
    plt.plot(x_ls, lk_ls)
    plt.xlabel(r"state $x$")
    plt.ylabel(r"stage cost $L(x, 0)$")
    plt.xlim([x_lb, x_ub])

    plt.grid(True)
    plt.tight_layout()
    if saveas is not None: plt.savefig(saveas, bbox_inches='tight', pad_inches=0.0)
    if show: plt.show()


def plot_traj(traj:Union[ScenarioTraj,List[ScenarioTraj]], show:bool=False, saveas:str=None, labels:Optional[List[str]]=None)-> None:

    if type(traj) is not list:
        traj = [traj]

    n_scen = [len(traj[i].x) for i in range(len(traj))]
    lw = [.7 if n > 1 else 1 for n in n_scen]
    alpha = [1 / (n**(1/3)) for n in n_scen]

    plt.figure(figsize=(3.5, 2.1))

    time_idx = np.arange(traj[0].x[0].shape[1])

    colors = ["C1", "C0"]
    plt.subplot(211)
    for i, tr in enumerate(traj):
        for x in tr.x:
            plt.plot(x.flatten(), color=colors[i], alpha=alpha[i], lw=lw[i])
    plt.plot([time_idx[0], time_idx[-1]],  2 * [-.1], '--', color='black')
    plt.ylabel(r"state $x$")
    plt.xlim([0, time_idx[-1]])
    plt.ylim([-.2, 1.1])
    plt.grid(True)
    plt.gca().set_xticklabels([])

    plt.subplot(212)
    for i, tr in enumerate(traj):
        for u in tr.u:
            plt.step(time_idx, np.concatenate((u, [[np.nan]]), axis=1).flatten(),  where='post', color=colors[i], alpha=alpha[i], lw=lw[i])
    plt.tight_layout()
    plt.ylabel(r"control $u$")
    plt.xlabel(r"discrete time $k$")
    plt.xlim([0, time_idx[-1]])
    plt.ylim([-3.3, .5])
    plt.grid(True)

    if labels is not None:
        lines = [Line2D([0], [0], label=labels[i], color=colors[i]) for i in range(len(traj))]
        plt.legend(handles=lines)
    plt.gcf().align_ylabels()
    plt.tight_layout()
    if saveas is not None: plt.savefig(saveas, bbox_inches='tight', pad_inches=0.0)
    if show: plt.show()


def plot_suboptimality_and_delta_u0(results: ResultsValFunc, saveas=None, show=False) -> None:

    N_xvals = results.x_vals.shape[0]

    sig_vals = np.abs(results.sig_vals)
    delta_u0 = results.u0_sub - results.u0_opt
    suboptimality = results.val_sub - results.val_opt

    markers = ['x', '+', 's', '1', '2']
    markers = ['x'] * 5
    ms = 5                                   # marker size

    lines = []
    labels = []

    plt.figure(figsize=(3, 2.6))
    plt.subplot(211)
    plot_tol = 5e-15
    for i in range(N_xvals):
        plt.title(r"Suboptimality $V_\sigma^{\mathrm{cec}}(x) - V_\sigma^\star(x)$")
        l_ ,= plt.plot(sig_vals[suboptimality[:,i]>=plot_tol], suboptimality[suboptimality[:,i]>=plot_tol, i], markers[i], ms=ms,markerfacecolor='none', label="$x={:4.1f}$".format(results.x_vals[i]))
        lines.append(l_)
        labels.append(f"$x={results.x_vals[i]}$")
    sig_lim = np.array([np.min(sig_vals)/1.5, np.max(sig_vals)*1.5  ])
    l_, = plt.plot(sig_lim, 4e3 * sig_lim**4, 'k--', label=r"$\mathcal{O}(\sigma^4)$")
    lines.append(l_)
    labels.append(r"$\mathcal{O}(\sigma^4)$")
    # fake for legend entry
    l_, = plt.plot([0],[0], 'k:', label=r"$\mathcal{O}(\sigma^2)$")
    lines.append(l_)
    labels.append(r"$\mathcal{O}(\sigma^2)$")
    plt.xlim(sig_lim)
    plt.xscale("log")
    plt.yscale("log")
    plt.gcf().legend(lines, labels, bbox_to_anchor=(.96, .5), loc ='center left', handlelength=1)
    plt.ylim([plot_tol, suboptimality.max() * 15])
    plt.grid(True)
    plt.gca().set_xticklabels([])

    plt.subplot(212)
    for i in range(N_xvals):
        plt.title(r"Policy difference $\Vert \pi^{\mathrm{cec}}(x) - \pi_\sigma^\star(x)\Vert$")
        plt.plot(sig_vals, np.abs(delta_u0[:,i]), 'x', ms=ms, label=f"$x={results.x_vals[i]}$")
    plt.plot(sig_lim, 1e1 * sig_lim**2, 'k:', label=r"$\mathcal{O}(\sigma^2)$")
    plt.xlim(sig_lim)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"standard deviation $\sigma$")
    plt.grid(True)

    plt.tight_layout()
    if saveas is not None: plt.savefig(saveas, bbox_inches='tight', pad_inches=0.0)
    if show: plt.show()


def plot_valfunc_and_policy(prob:treeOCP, results: ResultsValFunc, results_nom: Optional[ResultsValFunc]=None,
                            plot_stage_cost: bool=True, saveas=None, show=False) -> None:

    n_sig = results.sig_vals.shape[0]

    plt.figure(figsize=(3.5, 1.8))
    colors = ['C2', 'C0', 'C4', 'C3', 'C4']

    for i in reversed(range(n_sig)):
        sig_str = f"{results.sig_vals[i]:.2f}"
        sig_eval_str = r"\vert_{\sigma=" + sig_str + r"}"
        plt.plot(results.x_vals, results.val_sub[i,:], ':', color=colors[i], label=r"$V_\sigma^{\mathrm{cec}}(x)" + sig_eval_str + r"$") 
        plt.plot(results.x_vals, results.val_opt[i,:], '--', color=colors[i], label=r"$V_\sigma^\star(x)"          + sig_eval_str + r"$")

    if results_nom is not None:
        plt.plot(results_nom.x_vals, results_nom.val_opt, '-.', color=colors[n_sig], label=r"$V_0^\star(x) \equiv V_0^{\mathrm{cec}}(x)$")
    if plot_stage_cost:
        lk_vals = eval_stage_state_cost(prob, results.x_vals)
        plt.plot(results.x_vals, lk_vals, color=colors[n_sig+1], label=r"$L(x, 0)$")

    plt.xlim([np.min(results.x_vals), np.max(results.x_vals)])
    plt.ylim([-.1, 6])
    plt.grid(True)
    plt.xlabel(r"state $x$")
    plt.legend(bbox_to_anchor=(1.02, .5), loc='center left', handlelength=2)

    plt.tight_layout()
    if saveas is not None: plt.savefig(saveas, bbox_inches='tight', pad_inches=0.0)
    if show: plt.show()
