from typing import List, Tuple
import numpy as np
import casadi as ca
from solver_treeMPC import treeOCP
from policy_cec import PolicyTV


def eval_policy(prob: treeOCP, policy: PolicyTV, x: np.ndarray, p: np.ndarray) -> Tuple[float, List[List[np.ndarray]], List[List[np.ndarray]]]:

        f_func = ca.Function("f_func", [prob.x, prob.u, prob.w, prob.p], [prob.f_expr])
        lk_func = ca.Function("lk_func", [prob.x, prob.u, prob.p], [prob.lk_expr])
        lN_func = ca.Function("lN_func", [prob.x,  prob.p], [prob.lN_expr])

        # build scenario tree
        obj = 0
        x_tree = []
        u_tree = []
        dist_tree = []          # probability of each node
        x_tree.append([ x ])
        dist_tree.append([1])

        # iterate through stages
        for k in range(prob.N):
            u_tree.append([])
            x_tree.append([])
            dist_tree.append([])

            # iterate through scenarios
            l_k_sum = 0             # sum of stage costs for current stage
            for i in range(prob.n_dist**k):
                # control variable for current scenario
                xki = x_tree[-2][i]
                uki = policy.eval(xki, p, k)
                u_tree[-1].append(uki)

                # add stage cost
                l_k_sum += lk_func(xki, uki, p) * dist_tree[-2][i]

                # for each possible current state and control, apply all possible disturbance values
                for j in range(prob.n_dist):
                    x_next = f_func(xki, uki, prob.Wval[j], p).full()
                    x_tree[-1].append(x_next)
                    dist_tree[-1].append(prob.Wdist[j] * dist_tree[-2][i])

            l_k_sum *= prob.gamma**k        # discounting for current stage
            obj += l_k_sum

        # add terminal cost
        l_k_sum = 0
        for i in range(prob.n_dist**prob.N):
            l_k_sum += lN_func(x_tree[-1][i], p) * dist_tree[-1][i]
        l_k_sum *= prob.gamma**prob.N
        obj += l_k_sum
        obj = obj.full().squeeze()
        return obj, x_tree, u_tree