from tge import TGE
from utils import car, cadr, cdr, info
import numpy as np
import tensorflow as tf

def sample(logit, e=0):
    p = tf.math.sigmoid(logit)
    def f(x):
        if np.random.rand() < e:
            return np.random.choice(2)
        else:
            return int(np.random.rand() < x)
    return np.vectorize(f)(p)

def evaluate(record, ncclmask, nodemask):
    gdef = record["gdef"]
    strategy = { gdef.node[i].name: [int(ncclmask[i])] + [ int(nodemask[i, j]) for j in range(nodemask.shape[1]) ] for i in range(nodemask.shape[0]) }
    # strategy = { gdef.node[i].name: [int(ncclmask[gi])] + [ int(nodemask[j, gi]) for j in range(nodemask.shape[0]) ] for gi, group in enumerate(record["cgroups"]) for i in group }
    # info(strategy)
    leftout = [ gi for gi in range(nodemask.shape[1]) if np.sum(nodemask[:, gi]) == 0 ]
    for k, v in strategy.items():
        if np.sum(v[1:]) == 0:
            v[1] = 1
    tge = TGE(gdef, [dev for dev, _, _ in record["devices"]])
    tge.set_strategy(strategy)
    tge.fill_batchsize(120)
    tge.replace_placeholder(120)
    tge.set_bandwidth(intra=int(record["intra"]), inter=int(record["inter"]))
    tge.set_nccl_model(record["nccl_models"])
    time, mem = tge.evaluate(record["prof_data"])

    oom = [ i for i in range(len(mem)) if mem[i] > record["devices"][i][2] ]
    # return time / 1_000_000, [x / 1_000_000_000 for x in mem]
    return np.sqrt(time / 1_000_000), oom, leftout

def sample_and_evaluate(record, placement_logit):
    placement_mask = sample(nodelogit)
    sqrt_time, oom, leftout = evaluate(record, placement_mask)

    if 'hist' not in record:
        record["hist"] = []

    if len(oom) == 0 and len(leftout) == 0:
        record["hist"].append(sqrt_time)
        record["hist"] = record["hist"][-100:]
        baseline = np.mean(record["hist"])
        advantage = -(sqrt_time - baseline) / baseline
    else:
        advantage = 0

    return ncclmask, nodemask, advantage, sqrt_time, oom, leftout
