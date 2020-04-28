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

def evaluate(computation_graph, topo, decisions):
    gdef = computation_graph["gdef"]
    strategy = { gdef.node[i].name: [0, *decisions[i]] for i in range(decisions.shape[0]) }
    for k, v in strategy.items():
        if np.sum(v[1:]) == 0:
            v[1] = 1
    tge = TGE(gdef, [dev for dev, _ in topo["devices"]])
    tge.set_strategy(strategy)
    tge.fill_batchsize(48)
    tge.use_collective()
    tge.set_bandwidth(intra=topo["intra"], inter=topo["inter"])
    time, mem = tge.evaluate(computation_graph["prof_data"])

    return time
