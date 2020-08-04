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
    new_strategy[np.random.choice(len(s))] = np.random.choice(8)
    return new_strategy

def P(loss_old, loss_new, T):
    if loss_new <= loss_old:
        return 1
    else:
        return np.exp(-1 / T)

record = get_all_data()[8]

s, baseline = None, 9999
for i in range(8):
    decisions = [i for _ in range(len(record["cgroups"]))]
    loss = evaluate(record, decisions)
    if loss < baseline:
        s, baseline = decisions, loss

info(s)

loss = 1
best = 1
for epoch in range(10000):
    T = (epoch + 1) / 1000
    s_new = neighbour(s)
    loss_new = evaluate(record, s_new) / baseline
    if loss_new < best:
        best = loss_new
    if P(loss, loss_new, T) > np.random.rand():
        s, loss = s_new, loss_new
        info(epoch, loss, best)
