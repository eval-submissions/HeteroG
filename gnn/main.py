import numpy as np
import time
import tensorflow as tf
import pickle

from data import *
from model import GAT
from environment import sample, evaluate

import sys
def info(*args):
    print(*args, file=sys.stderr, flush=True)

(gdef, prof_dict) = pickle.load(open("vgg.pickle", "rb"))
topo1 = gen_topo([
    ("/job:worker/replica:0/task:0/device:GPU:0", 1),
    ("/job:worker/replica:0/task:0/device:GPU:1", 1),
    ("/job:worker/replica:0/task:0/device:GPU:2", 1),
    ("/job:worker/replica:0/task:0/device:GPU:3", 1),
    ("/job:worker/replica:0/task:1/device:GPU:0", 1.2),
    ("/job:worker/replica:0/task:1/device:GPU:1", 1.2),
    ("/job:worker/replica:0/task:1/device:GPU:2", 1.2),
    ("/job:worker/replica:0/task:1/device:GPU:3", 1.2),
])
data1 = get_data(gdef, prof_dict, topo1)
topo2 = gen_topo([
    ("/job:worker/replica:0/task:0/device:GPU:0", 1),
    ("/job:worker/replica:0/task:0/device:GPU:1", 1.2),
])
data2 = get_data(gdef, prof_dict, topo2)

with tf.device("/gpu:0"):
    model = GAT(4, 1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.002, epsilon=1e-8)

    # initialize graph
    dur = []
    for epoch in range(20):
        if epoch % 2 == 0:
            topo, data = topo1, data1
        else:
            topo, data = topo2, data2

        computation_features = tf.convert_to_tensor(data["features"], dtype=tf.float32)
        device_features = tf.convert_to_tensor(topo["features"], dtype=tf.float32)
        model.set_graphs(data["graph"], topo["graph"])

        if epoch >= 3:
            t0 = time.time()
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)
            logp = model([computation_features, device_features], training=True)
            mask, decisions = sample(logp.numpy()) # numpy to turn off gradient tracking
            loss_env = evaluate(data, topo, decisions)
            loss = loss_env * tf.reduce_sum(tf.boolean_mask(logp, mask))
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if epoch >= 3:
            dur.append(time.time() - t0)
            info(np.mean(dur))

        info(loss.numpy(), loss_env)
