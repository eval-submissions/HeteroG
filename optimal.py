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
    "/job:tge/replica:0/task:0/device:GPU:1"
)

with open("mlp.data", "r") as f:
    records = (x.strip().split(" ") for x in f.readlines())
    prof = {items[0]: [int(float(x)) for x in items[1:]] for items in records}

import tge

options = [[0, 1], [1, 0], [0, 2], [2, 0], [1, 1]]

import sys
dec = [int(x) for x in sys.argv[1].split('_')] # first two are 0 and 1, remainings are 0-4
ngiven = len(dec)
i = 2
j = 0

p = []
for node in gdef.node:
    if node.op in ('Const', 'Placeholder', 'NoOp', 'Assign'):
        p.append((0, [0, 1, 1]))
    elif node.op == 'ApplyGradientDescent':
        p.append((1, j))
        j += 1
    else:
        while len(dec) <= i:
            dec.append(0)
        p.append((2, i))
        i += 1

best = 2147483647

assert j == 2

while True:
    # test current
    d = {}
    for i, node in enumerate(gdef.node):
        if p[i][0] == 0:
            d[node.name] = p[i][1]
        elif p[i][0] == 1:
            d[node.name] = [dec[p[i][1]], 1, 1]
        else:
            d[node.name] = [0, *options[dec[p[i][1]]]]

    t = (tge.TGE(deepcopy(gdef), devices)
            .custom(d)
            .set_bandwidth(2000, 10000)
            .evaluate(prof))[0]

    if t < best:
        with open("best_{}.txt".format(sys.argv[1]), "w") as f:
            print(t, file=f)
            for i, x in enumerate(dec):
                if i < 2:
                    print(x, file=f)
                else:
                    print(options[x], file=f)
        best = t
        print("new best: {}".format(t))

    # next decision
    for i in range(ngiven, len(dec)):
        if dec[i] < len(options) - 1:
            dec[i] += 1
            dec[ngiven:i] = [0] * (i - ngiven)
            break
    else:
        print("all done")
        break

