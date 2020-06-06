from tge import TGE
import numpy as np

def sample(logp):
    mask = np.zeros(logp.shape, dtype=bool)
    result = np.zeros(logp.shape[:-1], dtype=int)
    for i in range(logp.shape[0]):
        for j in range(logp.shape[1]):
            s = int(np.random.choice(logp.shape[2], p=np.exp(logp[i, j, :])))
            mask[i, j, s] = 1
            result[i, j] = s
    return mask, result

def explore(logp):
    mask = np.zeros(logp.shape, dtype=bool)
    result = np.zeros(logp.shape[:-1], dtype=int)
    for i in range(logp.shape[0]):
        for j in range(logp.shape[1]):
            s = int(np.random.choice(logp.shape[2]))
            mask[i, j, s] = 1
            result[i, j] = s
    return mask, result

def evaluate(record, decisions):
    gdef = record["gdef"]
    strategy = { gdef.node[i].name: [2, *decisions[i]] for i in range(decisions.shape[0]) }
    penalty = 1
    for k, v in strategy.items():
        if np.sum(v[1:]) == 0:
            penalty += 1
            v[1] = 1
    tge = TGE(gdef, [dev for dev, _, _ in record["devices"]])
    tge.set_strategy(strategy)
    tge.fill_batchsize(48)
    tge.use_collective()
    tge.set_bandwidth(intra=record["intra"], inter=record["inter"])
    time, mem = tge.evaluate(record["prof_data"])

    for m, (_, limit, _) in zip(mem, record["devices"]):
        if m > limit:
            penalty += 1

    return (time / 1000000) * penalty ** .5

def sample_and_evaluate(record, logp):
    mask, decisions = sample(logp)
    loss = evaluate(record, decisions)
    return mask, loss

def explore_and_evaluate(record, logp):
    mask, decisions = explore(logp)
    loss = evaluate(record, decisions)
    return mask, loss

def evaluate_logp(record, logp, nsample=8, npool=4, nexplore=4, poolsize=100):
    if "pool" not in record:
        record["pool"] = []
    pool = record["pool"]
    results = []

    for _ in range(nsample):
        mask, loss = sample_and_evaluate(record, logp)
        results.append((mask, loss))
        if len(pool) < poolsize:
            pool.append((mask, loss))
        else: # randomly replace one in the pool
            i = np.random.choice(poolsize)
            if pool[i][1] > loss:
                pool[i] = mask, loss

    for _ in range(nexplore):
        mask, loss = explore_and_evaluate(record, logp)
        results.append((mask, loss))
        if len(pool) < poolsize:
            pool.append((mask, loss))
        else: # randomly replace one in the pool
            i = np.random.choice(poolsize)
            if pool[i][1] > loss:
                pool[i] = mask, loss

    for _ in range(npool):
        i = np.random.choice(len(pool))
        results.append(pool[i])

    return results
