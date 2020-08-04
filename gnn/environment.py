from tge import TGE
from utils import car, cadr, cdr
import numpy as np

import sys
def info(*args):
    print(*args, file=sys.stdout, flush=True)

def sample(logp, e=.05):
    mask = np.zeros(logp.shape, dtype='bool')
    d = []
    for i in range(logp.shape[0]):
        if np.random.rand() < e:
            nccl = np.random.choice(1)
            s = int(np.random.choice(logp.shape[1]))
        else:
            nccl = int(np.exp(logp[i, 1]) > np.random.rand())
            s = int(np.random.choice(logp.shape[1], p=np.exp(logp[i, 1:])))
        d.append((nccl, s))
        mask[i, 0] = nccl
        mask[i, 1+s] = 1
    return mask, d

def evaluate(record, decisions):
    decision_map = [ [ 1 if i in group else 0 for i in range(len(record["devices"])) ] for group in record["tgroups"] ]

    gdef = record["gdef"]
    if record["cgroups"] is not None:
        strategy = { gdef.node[i].name: [2 if decisions[gi][0] != 0 else 0] + decision_map[decisions[gi][1]] for gi, group in enumerate(record["cgroups"]) for i in group }
    else:
        strategy = { gdef.node[i].name: [2 if decisions[i][0] != 0 else 0] + decision_map[decisions[i][1]] for i in range(len(decisions)) }
    penalty = 1
    # for k, v in strategy.items():
    #     if np.sum(v[1:]) == 0:
    #         penalty += 1
    #         v[1] = 1
    tge = TGE(gdef, [dev for dev, _, _ in record["devices"]])
    tge.set_strategy(strategy)
    tge.fill_batchsize(48)
    tge.replace_placeholder(48)
    tge.set_bandwidth(intra=int(record["intra"]), inter=int(record["inter"]))
    tge.set_nccl_model(record["nccl_models"])
    time, mem = tge.evaluate(record["prof_data"])

    for m, (_, _, limit) in zip(mem, record["devices"]):
        # info(m, limit)
        if m > limit:
            penalty += 1

    return (time / 1000000) * penalty ** .5

def sample_and_evaluate(record, logp):
    mask, decisions = sample(logp)
    loss = evaluate(record, decisions)
    return mask, loss

def evaluate_logp(record, logp):
    if 'best_single' not in record:
        best_single = None, 9999
        for nccl in range(1):
            for i in range(logp.shape[1]):
                decisions = [ [nccl, i] for _ in range(logp.shape[0]) ]
                loss = evaluate(record, decisions)
                if loss < cadr(best_single):
                    mask = np.zeros(logp.shape, dtype=bool)
                    for i in range(logp.shape[0]):
                        mask[i, 0] = decisions[i][0]
                        mask[i, decisions[i][1]] = 1
                    best_single = mask, loss
        info(best_single[0][0])
        record["best_single"] = best_single

    mask, loss = sample_and_evaluate(record, logp)
    baseline = cadr(record["best_single"])

    if 'best' not in record or loss < cadr(record['best']):
        record["best"] = mask, loss

    return mask, (loss - baseline) / baseline
