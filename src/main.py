import os
import numpy as np
from solver_treeMPC import SolverTreeMPC
from policy_cec import PolicyCEC
from utils_save_load import ResultsValFunc
from simple_prob import set_up_problem
from simple_prob_plotutils import set_plot_defaults, plot_traj, plot_suboptimality_and_delta_u0, plot_stage_state_cost, plot_valfunc_and_policy
from experiments import run_experiment_subopt, solve_OCP

# If true: run experiments and save to resultsfolder
# Else: only load from results from resultsfolder
run_experiments = True

plotfolder = "../plots/"
plotfolder = set_plot_defaults(plotfolder)

resultsfolder = "../results/"
if not os.path.exists(resultsfolder):
    os.makedirs(resultsfolder)

#%% set up problem and solvers
x = 1
sig = 1e-1

prob_stoch = set_up_problem()
solver_stoch = SolverTreeMPC(prob_stoch)

prob_nom = set_up_problem(noise_dist='no_noise')
solver_nom = SolverTreeMPC(prob_nom)

#%% solve and visualize OCP, nominal and stochastic
traj_nom = solve_OCP(solver_nom, x, 0)
traj_stoch = solve_OCP(solver_stoch, x, sig)

plot_traj(traj_stoch,saveas=plotfolder + "traj_stoch.pdf")
plot_traj(traj_nom, saveas=plotfolder + "traj_nom.pdf")
plot_traj([traj_stoch, traj_nom], saveas=plotfolder + "traj_nom_stoch.pdf", labels=[r"$\sigma = 0.1$", r"$\sigma = 0$"])
plot_stage_state_cost(prob_nom, saveas=plotfolder + "stage_cost_state.pdf", show=False)

#%% compute delta u0 and suboptimality of CEC as function of sigma, for several values of x
policy_cec = PolicyCEC(prob_nom)
sig_vals = np.logspace(-6, -.5, 20)
x_vals = [-.2, -.1, 0, .5, 1]

if run_experiments:
    results = run_experiment_subopt(sig_vals, x_vals, prob_stoch, policy_cec, solver_stoch, saveas=resultsfolder + "subopt_many_xval.pkl")
results = ResultsValFunc.load(resultsfolder + "subopt_many_xval.pkl")
plot_suboptimality_and_delta_u0(results, saveas=plotfolder + "suboptimality_and_delta_u0.pdf", show=False)

#%% compute value functions
x_vals = np.linspace(-.3, .5, 100)
sig_vals = [.05, .1]

if run_experiments:
    results_nom = run_experiment_subopt([0], x_vals, None, None, solver_opt=solver_nom, eval_subopt=False, saveas=resultsfolder + "valfunc_nom.pkl")
    results     = run_experiment_subopt(sig_vals, x_vals, prob_stoch, policy_cec, solver_stoch, saveas=resultsfolder + "valfunc_several_sig.pkl")
results_nom = ResultsValFunc.load(resultsfolder + "valfunc_nom.pkl")
results     = ResultsValFunc.load(resultsfolder + "valfunc_several_sig.pkl")
plot_valfunc_and_policy(prob_stoch, results, results_nom=results_nom, plot_stage_cost=True, saveas=plotfolder + "valfunc.pdf", show=False)

