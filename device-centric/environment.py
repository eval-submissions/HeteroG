from tge import TGE
from utils import car, cadr, cdr
import numpy as np

import sys
def info(*args):
    print(*args, file=sys.stdout, flush=True)

def sample(logp, e=.05):
    def f(x):
        if np.random.rand() > e:
            return np.random.choice(2)
        else:
            return int(np.exp(x) > np.random.rand())
    return np.vectorize(f)(logp)

def evaluate(record, ncclmask, mask):
    gdef = record["gdef"]
    strategy = { gdef.node[i].name: [int(ncclmask[gi])] + [ int(mask[j, gi]) for j in range(mask.shape[0]) ] for gi, group in enumerate(record["cgroups"]) for i in group }
    # info(strategy)
    penalty = 1
    for k, v in strategy.items():
        if np.sum(v[1:]) == 0:
            penalty += 1
            v[1] = 1
    tge = TGE(gdef, [dev for dev, _, _ in record["devices"]])
    tge.set_strategy(strategy)
    tge.fill_batchsize(120)
    tge.replace_placeholder(120)
    tge.set_bandwidth(intra=int(record["intra"]), inter=int(record["inter"]))
    tge.set_nccl_model(record["nccl_models"])
    time, mem = tge.evaluate(record["prof_data"])

    for m, (_, _, limit) in zip(mem, record["devices"]):
        # info(m, limit)
        if m > limit:
            penalty += 1

    return (time / 1000000) * penalty ** .5

def sample_and_evaluate(record, nccllogp, logp):
    mask = sample(logp)
    ncclmask = sample(nccllogp)
    loss = evaluate(record, ncclmask, mask)
    return ncclmask, mask, loss

def evaluate_logp(record, nccllogp, logp):
    ncclmask, mask, loss = sample_and_evaluate(record, nccllogp, logp)

    if 'best' not in record or loss < record['best'][2]:
        record["best"] = ncclmask, mask, loss

    return ncclmask, mask, loss
