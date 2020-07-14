import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import tensorflow as tf
import sys
import pickle

from data import get_all_data
from environment import evaluate
from tge import TGE

def derive_new_strategy(old, diff):
    result = []
    for o, d in zip(old, diff):
        n = [x for x in o]
        for i in range(len(d)):
            if d[i] == 1:
                if n[i] < 2:
                    n[i] += 1
            elif d[i] == -1:
                if n[i] > 0:
                    n[i] -= 1
        result.append(n)
    return result

def loss_fn(strategy):
    decisions = np.array([strategy[groups[i]] for i in range(len(record['gdef'].node))])
    return evaluate(record, decisions)

record = get_all_data()[int(sys.argv[1])]
devices = [name for name, _, _ in record['devices']]
tge = TGE(record['gdef'], devices)
groups = tge.get_groups()

pool = []
for s in (
    [[1] * len(devices) for i in range(max(groups) + 1)],
    [[2] * len(devices) for i in range(max(groups) + 1)],
    [[1] + [0] * (len(devices) - 1) for i in range(max(groups) + 1)],
):
    pool.append((s, loss_fn(s)))

for epoch in range(100):
    s, loss = pool[np.random.choice(len(pool))]
    for trial in range(100):
        random_diff = [[np.random.randint(-1, 2) * (np.random.rand() < .1) for i in range(len(g))] for g in s]
        new_strategy = derive_new_strategy(s, random_diff)
        new_loss = loss_fn(new_strategy)
        if new_loss < loss:
            pool.append((new_strategy, new_loss))
            if len(pool) > 100:
                pool = [(s, l) for s, l in pool if l < loss]
    if epoch % 10 == 0:
        record["pool"] = pool
        with open("search{}.pickle".format(sys.argv[1]), 'wb') as x:
            pickle.dump(record, x)
