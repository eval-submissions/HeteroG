import tensorflow as tf
import xgboost as xgb
import numpy as np
import pickle
from tge import TGE

vgg_model, vgg_prof = pickle.load(open("vgg.pickle", 'rb'))
resnet_model, resnet_prof = pickle.load(open("vgg.pickle", 'rb'))
lenet_model, lenet_prof = pickle.load(open("vgg.pickle", 'rb'))
mlp_model, mlp_prof = pickle.load(open("vgg.pickle", 'rb'))

BATCHSIZE = 48
MAX_REPLICA = 4
DEVICES = (
    "/job:worker/replica:0/task:0/device:GPU:0",
    "/job:worker/replica:0/task:0/device:GPU:1"
)

def random_strategy(model):
    return [[1, 1, 1] for i in range(max(TGE(model, DEVICES).get_groups()) + 1)]

def derive_new_strategy(old, diff):
    result = []
    for o, d in zip(old, diff):
        for i in range(len(d)):
            if d[i] == 1:
                if sum(o[1:]) >= MAX_REPLICA:
                    # already reach maximum, ignore
                    continue
                o[i+1] += 1
            elif d[i] == -1:
                if o[i+1] <= 0:
                    # no replica here, ignore
                    continue
                elif sum(o[1:]) <= 1:
                    # no replica left, ignore
                    continue
                o[i+1] -= 1
        result.append(o)
    return result

def train(data):
    pass

# loss: (OOM, time). It compares naturally in Python since True > False
def simulate(model, prof, strategy):
    tge = TGE(model, DEVICES)
    groups = tge.get_groups()
    tge.custom({ node.name: strategy[groups[i]] for (i, node) in enumerate(model.node) })
    tge.replace_placeholder(BATCHSIZE)
    tge.use_collective()
    tge.compile()
    time, memory = tge.evaluate(prof)
    OOM = any((m > 10 * 2 ** 30 for m in memory))
    return (OOM, time), []

strategy = random_strategy(vgg_model)
loss, meta = simulate(vgg_model, vgg_prof, strategy)
print(loss)
for epoch in range(100):
    random_diff = [[np.random.randint(-1, 2) for i in range(len(g) - 1)] for g in strategy]
    new_strategy = derive_new_strategy(strategy, random_diff)
    new_loss, _ = simulate(vgg_model, vgg_prof, strategy)
    print(new_loss)
    if new_loss < loss:
        loss = new_loss
        strategy = new_strategy

print(loss)
print(strategy)
