import numpy as np
import time
import tensorflow as tf

from copy import copy

from data import get_all_data
from model import Model
from environment import evaluate_logp, evaluate
from utils import save, load

import sys
def info(*args):
    print(*args, file=sys.stdout, flush=True)

def neighbour(strategy):
    new_strategy = copy(s)
    new_strategy[np.random.choice(len(s))] = np.random.choice(1), np.random.choice(8)
    return new_strategy

def P(loss_old, loss_new, T):
    if loss_new <= loss_old:
        return 1
    else:
        return np.exp(1 - 1 / T)

record = get_all_data()[2]

# decisions = [ [1, 7] for _ in range(len(record["cgroups"])) ]
# evaluate(record, decisions)
# sys.exit(0)

s, baseline = None, 9999
for nccl in range(2):
    for i in range(8):
        decisions = [ [nccl, i] for _ in range(len(record["cgroups"])) ]
        loss = evaluate(record, decisions)
        info(decisions, loss)
        if loss < baseline:
            s, baseline = decisions, loss

loss = 1
best = 1
for epoch in range(20000):
    T = (0.5 * epoch + 10000) / 20000
    s_new = neighbour(s)
    loss_new = evaluate(record, s_new) / baseline
    if loss_new < best:
        best = loss_new
    if P(loss, loss_new, T) > np.random.rand():
        s, loss = s_new, loss_new
        info(epoch, loss, best)
