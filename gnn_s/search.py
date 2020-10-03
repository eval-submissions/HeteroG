import tensorflow as tf
import numpy as np
from environment import sample, evaluate, sample_and_evaluate
from utils import save, load, info

from pymoo.model.problem import Problem

class MyProblem(Problem):
    def __init__(self, record):
        self.record = record
        n = len(record['cgroups']) * len(record['devices'])
        super().__init__(n_var=n, n_obj=1, n_constr=0, xl=0, xu=1, elementwise_evaluation=True)

    def _evaluate(self, x, out, *args, **kwargs):
        pheno = (x > .5).astype(int)
        strategy = np.reshape(pheno, (len(self.record['cgroups']), len(self.record['devices'])))
        k = evaluate(self.record, [1] * len(self.record['cgroups']), strategy)[0]
        info(k)
        out["F"] = k
        out["pheno"] = pheno
        out["hash"] = hash(str(pheno))

record = load("records")[7] # 15
problem = MyProblem(record)

from pymoo.algorithms.so_brkga import BRKGA
from pymoo.optimize import minimize

from pymoo.model.duplicate import ElementwiseDuplicateElimination

class MyElementwiseDuplicateElimination(ElementwiseDuplicateElimination):
    def is_equal(self, a, b):
        info(a.get("hash"))
        return a.get("hash") == b.get("hash")

from pymoo.model.sampling import Sampling

class MySampling(Sampling):
    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), None, dtype=np.float)

        for i in range(n_samples):
            for j in range(problem.n_var):
                X[i, j] = 1 if np.random.rand() < .95 else 0

        return X

algorithm = BRKGA(
    n_elites=20,
    n_offsprings=60,
    n_mutants=20,
    bias=0.7,
    sampling=MySampling(),
    eliminate_duplicates=MyElementwiseDuplicateElimination)

res = minimize(problem,
               algorithm,
               ("n_gen", 20),
               seed=1,
               verbose=False)

info("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
info("Solution", res.opt.get("pheno")[0])

# def search(record):
