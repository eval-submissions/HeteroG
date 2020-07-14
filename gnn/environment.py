from tge import TGE
import numpy as np

import sys
def info(*args):
    print(*args, file=sys.stdout, flush=True)

def sample(logp):
    mask = np.zeros(logp.shape, dtype='bool')
    d = []
    for i in range(logp.shape[0]):
        s = int(np.random.choice(logp.shape[1], p=np.exp(logp[i])))
        d.append(s)
        mask[i, s] = 1
    return mask, d

def evaluate(record, decisions):
    decision_map = [
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 1, 1, 1, 1, 0, 0],
        [2, 1, 1, 1, 1, 0, 0]
    ]

    gdef = record["gdef"]
    if record["groups"] is not None:
        strategy = { gdef.node[i].name: decision_map[decisions[gi]] for gi, group in enumerate(record["groups"]) for i in group }
    else:
        strategy = { gdef.node[i].name: decision_map[decisions[i]] for i in range(len(decisions)) }
    penalty = 1
    for k, v in strategy.items():
        if np.sum(v[1:]) == 0:
            penalty += 1
            v[1] = 1
    tge = TGE(gdef, [dev for dev, _, _ in record["devices"]])
    tge.set_strategy(strategy)
    tge.fill_batchsize(48)
    # tge.use_collective()
    tge.set_bandwidth(intra=record["intra"], inter=record["inter"])
    time, mem = tge.evaluate(record["prof_data"])

    for m, (_, limit, _) in zip(mem, record["devices"]):
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
    if 'hist' not in record:
        loss = 9999999
        for i in range(8):
            d = np.zeros(logp.shape[0], dtype=int)
            d[:] = i
            l = evaluate(record, d)
            if l < loss:
                loss = l

        record['hist'] = [loss]

    if np.random.rand() < .9:
        mask, loss = sample_and_evaluate(record, logp)
    else:
        d = np.zeros(logp.shape[0], dtype=int)
        loss = evaluate(record, d)
        mask = np.zeros(logp.shape, dtype=bool)
        for i in range(logp.shape[0]):
            mask[i, d[i]] = 1

    avg = sum(record['hist']) / len(record['hist'])
    if loss < avg:
        record['hist'].append(loss)

    return mask, (loss - avg) / avg
