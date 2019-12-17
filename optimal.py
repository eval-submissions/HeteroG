import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import tensorflow as tf
from copy import deepcopy

gdef = tf.get_default_graph().as_graph_def()

with open("mlp.model", "rb") as f:
    gdef.ParseFromString(f.read())

devices = (
    "/job:tge/replica:0/task:0/device:GPU:0",
    "/job:tge/replica:0/task:0/device:GPU:1",
    "/job:tge/replica:0/task:1/device:GPU:0",
    "/job:tge/replica:0/task:1/device:GPU:1"
)

with open("mlp.data", "r") as f:
    records = (x.strip().split(" ") for x in f.readlines())
    prof = {items[0]: [int(float(items[1]))] * len(devices) for items in records}

import tge

options = set()
for x1 in range(5):
    for x2 in range(5):
        for x3 in range(5):
            for x4 in range(5):
                if 0 < x1 + x2 + x3 + x4 <= 4:
                    options.add((x1, x2, x3, x4))
options = list(options)

dec = []
bdec = 0

p = []
for node in gdef.node:
    if node.op in ('Const', 'Placeholder', 'NoOp', 'Assign'):
        p.append((0, [0, 1, 1, 1, 1]))
    elif node.op == 'ApplyGradientDescent':
        p.append((1, bdec))
        bdec += 1
    else:
        i = len(dec)
        dec.append(0)
        p.append((2, i))

best = 2147483647

assert bdec == 3

import sys
bdec = [int(x) for x in sys.argv[1].split('_')]
assert len(bdec) == 3

while True:
    # test current
    d = {}
    for i, node in enumerate(gdef.node):
        if p[i][0] == 0:
            d[node.name] = p[i][1]
        elif p[i][0] == 1:
            d[node.name] = [bdec[p[i][1]], 1, 1, 1, 1]
        else:
            d[node.name] = [0, *options[dec[p[i][1]]]]

    t = (tge.TGE(deepcopy(gdef), devices)
            .custom(d)
            .set_bandwidth(2000, 10000)
            .evaluate(prof))[0]

    if t < best:
        with open("best_{}.txt".format(sys.argv[1]), "w") as f:
            print(t, file=f)
            for x in dec:
                print(options[x], file=f)
            for x in bdec:
                print(x, file=f)
        best = t
        print("new best: {}".format(t))

    # next decision
    for i in range(len(dec)):
        if dec[i] < len(options) - 1:
            dec[i] += 1
            dec[:i] = [0] * i
            break
    else:
        print("all done")
        break

