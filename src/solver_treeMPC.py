from typing import List
from dataclasses import dataclass
import numpy as np
import casadi as ca
from suppress_output import suppress_stderr

@dataclass
class treeOCP:
    x: ca.SX                        # state symbol
    u: ca.SX                        # control symbol
    p: ca.SX                        # parameter symbol
    w: ca.SX                        # disturbance symbol
    f_expr: ca.SX                   # dynamics expression
    lk_expr: ca.SX                  # stage cost expression
    lN_expr: ca.SX                  # terminal cost expression
    N: int                          # prediction horizon
    Wval: List[np.ndarray]          # disturbance values per stage
    Wdist: List[float]              # disturbance probabilities per stage
    gamma: float = 1                # discount factor

    @property
    def nx(self) -> int:
        return self.x.shape[0]
    @property
    def nu(self) -> int:
        return self.u.shape[0]
    @property
    def np(self) -> int:
        return self.p.shape[0]
    @property
    def nw(self) -> int:
        return self.w.shape[0]
    @property
    def n_dist(self) -> int:           # number of disturbance values
        return len(self.Wval)

@dataclass
class ScenarioTraj:
    x: List[np.ndarray]
    u: List[np.ndarray]


class SolverTreeMPC():
    def __init__(self, problem: treeOCP):
        self.problem = problem
        self._build_solver()
        self.success = False
        self._sol = None


    def _build_solver(self) -> None:

        f_func = ca.Function("f_func", [self.problem.x, self.problem.u, self.problem.w, self.problem.p], [self.problem.f_expr])
        lk_func = ca.Function("lk_func", [self.problem.x, self.problem.u, self.problem.p], [self.problem.lk_expr])
        lN_func = ca.Function("lN_func", [self.problem.x,  self.problem.p], [self.problem.lN_expr])

        # build tree OCP
        obj = 0
        x_tree = []
        u_tree = []
        dist_tree = []          # probability of each node
        g = []
        lbg = []
        ubg = []
        x_tree.append([ ca.SX.sym("x_0", self.problem.nx) ])
        dist_tree.append([1])

        # iterate through stages
        for k in range(self.problem.N):
            u_tree.append([])
            x_tree.append([])
            dist_tree.append([])
            # iterate through scenarios
            l_k_sum = 0             # sum of stage costs for current stage
            for i in range(self.problem.n_dist**k):
                # control variable for current scenario
                uki = ca.SX.sym(f"u_{k}_{i}", self.problem.nu)
                u_tree[-1].append(uki)
                # add stage cost
                l_k_sum += lk_func(x_tree[-2][i], uki, self.problem.p) * dist_tree[-2][i]

                # for each possible current state and control, apply all possible disturbance values
                for j in range(self.problem.n_dist):
                    x_next = f_func(x_tree[-2][i], uki, self.problem.Wval[j], self.problem.p)
                    xkij = ca.SX.sym(f"x_{k+1}_{i*self.problem.n_dist + j}", self.problem.nx)
                    x_tree[-1].append(xkij)             
                    dist_tree[-1].append(self.problem.Wdist[j] * dist_tree[-2][i]) 
                    # add dynamics constraints
                    g.append(xkij - x_next)
                    lbg.append(np.zeros(self.problem.nx))
                    ubg.append(np.zeros(self.problem.nx))

            l_k_sum *= self.problem.gamma**k        # discounting for current stage
            obj += l_k_sum 

        # add terminal cost
        l_k_sum = 0
        for i in range(self.problem.n_dist**self.problem.N):
            l_k_sum += lN_func(x_tree[-1][i], self.problem.p) * dist_tree[-1][i]
        l_k_sum *= self.problem.gamma**self.problem.N
        obj += l_k_sum

        p = self.problem.p
        decvar_ = []
        for k in range(self.problem.N):
            decvar_.extend(x_tree[k])
            decvar_.extend(u_tree[k])
        decvar_.extend(x_tree[self.problem.N])
        decvar = ca.veccat(*decvar_)
        g = ca.veccat(*g)

        nlp = {"x": decvar, "f": obj, "g": g, "p": p}
        opts = {}
        opts["print_time"] = 0
        opts["verbose"] = False
        opts["ipopt"] = {}
        opts["ipopt"]["print_level"] = 0
        opts["ipopt"]["sb"] = "yes"
        opts["ipopt"]["tol"] = 1e-12
        self._solver = ca.nlpsol("solver", "ipopt", nlp, opts)
        self._decvar = decvar
        self._lbg = np.concatenate(lbg)
        self._ubg = np.concatenate(ubg)
        self._lbx = -np.inf * np.ones(decvar.shape)
        self._ubx = np.inf * np.ones(decvar.shape)
        self._x_tree = x_tree
        self._u_tree = u_tree
        self._dist_tree = dist_tree


    def solve(self, x0, p) -> np.ndarray:

        self._lbx[:self.problem.nx] = x0
        self._ubx[:self.problem.nx] = x0
        with suppress_stderr():     # brute force silencing of casadi / ipopt
            if self._sol is None:
                self._sol = self._solver(p=p, lbx=self._lbx, ubx=self._ubx, lbg=self._lbg, ubg=self._ubg)
            else:
                self._sol = self._solver(x0=self._sol["x"], p=p, lbx=self._lbx, ubx=self._ubx, lbg=self._lbg, ubg=self._ubg)

        return_status = self._solver.stats()["return_status"]
        self.success = self._solver.stats()["success"]

        # also catches minor things like "solved to acceptable level"
        if return_status != "Solve_Succeeded":
            print("return status:", return_status)

        # if the solver fully fails
        if not self.success:
            print("solver failed")
            return np.nan * np.ones(self.problem.nu)

        return self.u0

    @property
    def u0(self) -> np.ndarray:
        if not self.success:
            print("solver not converged, returning nan")
            print(self._solver.stats()["return_status"])
            return np.nan * np.ones(self.problem.nu)
        return self._sol["x"][self.problem.nx:self.problem.nx+self.problem.nu].full().flatten()
    
    @property
    def obj_val(self) -> float:
        if not self.success:
            print("solver not converged, returning nan")
            print(self._solver.stats()["return_status"])
            return np.nan
        return self._sol["f"].full().squeeze()
    
    @property
    def x_tree(self) -> List[List[np.ndarray]]:
        return self._eval_tree_at_sol(self._x_tree)
    
    @property
    def u_tree(self) -> List[List[np.ndarray]]:
        return self._eval_tree_at_sol(self._u_tree)

    def _eval_tree_at_sol(self, SX_tree: List[List[ca.SX]]) -> List[List[np.ndarray]]:
        return [[ca.evalf(ca.substitute(yki, self._decvar, self._sol["x"])).full().flatten() for yki in yk_list ] for yk_list in SX_tree]

    def get_x_scenarios(self) -> List[np.ndarray]:
        return tree_to_scenario(self.x_tree)
    
    def get_u_scenarios(self) -> List[np.ndarray]:
        return tree_to_scenario(self.u_tree)
    
    def get_scenarios(self) -> ScenarioTraj:
        return ScenarioTraj(self.get_x_scenarios(), self.get_u_scenarios())

    def print_probablities_tree(self) -> None:
        print("Tree: probability of each node")
        for k in range(self.problem.N+1):
            outstr = 'stage {}: '.format(k)
            for probab in self._dist_tree[k]:
                outstr += "{} ,".format(probab)
            print(outstr)


def tree_to_scenario(tree: List[List[np.ndarray]]) -> List[np.ndarray]:

    N = len(tree)
    m = len(tree[1])
    n_scen = m**(N-1)

    scenarios = []
    for i in range(n_scen):
        scen = np.zeros((tree[0][0].shape[0], N))
        for k in range(N):
            scen[:, k] = tree[k][i // (m**(N-k-1) )]
        scenarios.append(scen)

    return scenarios
