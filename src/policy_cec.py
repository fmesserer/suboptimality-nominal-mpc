from solver_treeMPC import treeOCP, SolverTreeMPC

class PolicyTV():
    """
    Template for time varying policy
    """
    def __init__(self):
        pass

    def eval(self, x, p, k):
        pass


class PolicyCEC(PolicyTV):
    """
    Certainty equivalent control aka shrinking horizon nominal MPC
    """
    def __init__(self, prob: treeOCP):
        self._prob = prob
        self._pol_list = [None] * prob.N
        self.create()

    def create(self):
        for k in range(1, self._prob.N + 1):
            self._prob.N = k
            self._pol_list[-k] = SolverTreeMPC(self._prob)

    def eval(self, x, p, k):
        return self._pol_list[k].solve(x, p)
