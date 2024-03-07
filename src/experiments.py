import numpy as np
from typing import Optional, Union, List
from solver_treeMPC import SolverTreeMPC, treeOCP
from policy_cec import PolicyTV
from utils_save_load import ResultsValFunc
from eval_policy import eval_policy


def run_experiment_subopt(sig_vals: Union[List, np.ndarray], x_vals: Union[List, np.ndarray], prob_stoch: treeOCP, policy_subopt: PolicyTV,
                            eval_opt: bool = True, eval_subopt: bool = True,
                            solver_opt: Optional[SolverTreeMPC]=None,  saveas:Optional[str]=None) -> ResultsValFunc:

    if solver_opt is None:
        solver_opt = SolverTreeMPC(prob_stoch)
    if isinstance(x_vals, list):
        x_vals = np.array(x_vals)
    if isinstance(sig_vals, list):
        sig_vals = np.array(sig_vals)

    N_sig = sig_vals.shape[0]
    N_x = x_vals.shape[0]

    u0_opt = np.zeros((N_sig, N_x))     * np.nan
    u0_subopt = np.zeros((N_sig, N_x))  * np.nan
    val_opt = np.zeros((N_sig, N_x))    * np.nan
    val_subopt = np.zeros((N_sig, N_x)) * np.nan

    for i in range(sig_vals.shape[0]):
        for j in range(x_vals.shape[0]):
            print(f"sigma = {sig_vals[i]}, x = {x_vals[j]}")
            if eval_opt:
                print("solving stochastic problem...")
                u0_opt[i,j] = solver_opt.solve(x_vals[j] , sig_vals[i])
                val_opt[i,j] = solver_opt.obj_val
            if eval_subopt:
                print("evaluating policy...")
                val_subopt[i,j], _, u_tree_subopt = eval_policy(prob_stoch, policy_subopt, x_vals[j], sig_vals[i])
                u0_subopt[i,j] = u_tree_subopt[0][0]

    results = ResultsValFunc(x_vals, sig_vals, u0_opt.squeeze(), u0_subopt.squeeze(), val_opt.squeeze(), val_subopt.squeeze())
    if saveas is not None:
        results.save(saveas)
    return results


def solve_OCP(solver: SolverTreeMPC, x, p):

    u0 = solver.solve(x, p)
    traj = solver.get_scenarios()
    return traj

