from tge import TGE
import numpy as np

import sys
def info(*args):
    print(*args, file=sys.stdout, flush=True)

def sample(logp, e=0):
    mask = np.zeros(logp.shape, dtype='bool')
    d = []
    for i in range(logp.shape[0]):
        if np.random.rand() < e:
            s = int(np.random.choice(logp.shape[1]))
        else:
            s = int(np.random.choice(logp.shape[1], p=np.exp(logp[i])))
        d.append(s)
        mask[i, s] = 1
    return mask, d

def evaluate(record, decisions):
    decision_map = [
        [2 if len(group) >= 2 else 0] + [1 if i in group else 0 for i in range(len(record["devices"]))]
        for group in record["tgroups"]
    ]

    gdef = record["gdef"]
    if record["cgroups"] is not None:
        strategy = { gdef.node[i].name: decision_map[decisions[gi]] for gi, group in enumerate(record["cgroups"]) for i in group }
    else:
        strategy = { gdef.node[i].name: decision_map[decisions[i]] for i in range(len(decisions)) }
    penalty = 1
    # for k, v in strategy.items():
    #     if np.sum(v[1:]) == 0:
    #         penalty += 1
    #         v[1] = 1
    tge = TGE(gdef, [dev for dev, _, _ in record["devices"]])
    tge.set_strategy(strategy)
    tge.fill_batchsize(48)
    # tge.use_collective()
    tge.set_bandwidth(intra=record["intra"], inter=record["inter"])
    time, mem = tge.evaluate(record["prof_data"])

    for m, (_, _, limit) in zip(mem, record["devices"]):
        # info(m, limit)
        if m > limit:
            penalty += 1

    return (time / 1000000) * penalty ** .5

def predict_and_evaluate(record, logp):
    decisions = np.argmax(logp, axis=1)
    loss = evaluate(record, decisions)
    mask = np.zeros(logp.shape, dtype=bool)
    for i in range(logp.shape[0]):
        mask[i, decisions[i]] = 1
    return mask, loss

def sample_and_evaluate(record, logp):
    mask, decisions = sample(logp)
    loss = evaluate(record, decisions)
    return mask, loss

def evaluate_logp(record, logp):
    if 'pool' not in record:
        record['pool'] = []
        for i in range(logp.shape[1]):
            decisions = np.zeros(logp.shape[0], dtype=int)
            decisions[:] = i
            loss = evaluate(record, decisions)
            mask = np.zeros(logp.shape, dtype=bool)
            for i in range(logp.shape[0]):
                mask[i, decisions[i]] = 1
            record['pool'].append((mask, loss))

    pool = record['pool']

    i = np.random.choice(range(len(pool)))
    if np.random.rand() < .05:
        mask, loss = pool[i]
    else:
        mask, loss = sample_and_evaluate(record, logp)
        if loss < pool[i][1]:
            pool[i] = mask, loss
    avg = sum(l for _, l in pool) / len(pool)

    # j = 0
    # for i in range(len(pool)):
    #     if pool[i][1] < pool[j][1]:
    #         j = i

    # info(pool[j])

    return mask, (loss - avg) / avg
