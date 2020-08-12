from tge import TGE
from utils import car, cadr, cdr
import numpy as np
import tensorflow as tf

import sys
def info(*args):
    print(*args, file=sys.stdout, flush=True)

def sample(logit, e=.05):
    p = tf.math.sigmoid(logit)
    def f(x):
        if np.random.rand() < e:
            return np.random.choice(2)
        else:
            return int(np.random.rand() < x)
    return np.vectorize(f)(p)

def evaluate(record, ncclmask, nodemask):
    gdef = record["gdef"]
    strategy = { gdef.node[i].name: [int(ncclmask[gi])] + [ int(nodemask[j, gi]) for j in range(nodemask.shape[0]) ] for gi, group in enumerate(record["cgroups"]) for i in group }
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

    return np.sqrt(time / 1_000_000) * (10 if penalty >= 2 else 1) # penalty ** .5

def sample_and_evaluate(record, nccllogit, nodelogit):
    ncclmask = sample(nccllogit)
    nodemask = sample(nodelogit)
    loss = evaluate(record, ncclmask, nodemask)

    if 'best' not in record or loss < record['best'][2]:
        record["best"] = ncclmask, nodemask, loss

    return ncclmask, nodemask, loss

