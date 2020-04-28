import numpy as np
import time
import tensorflow as tf
import pickle

from data import *
from model import GAT
from environment import evaluate

(gdef, prof_dict) = pickle.load(open("vgg.pickle", "rb"))
topo = gen_topo([
    ("/job:worker/replica:0/task:0/device:GPU:0", 1),
    ("/job:worker/replica:0/task:0/device:GPU:1", 1.1)
])
data = get_data(gdef, prof_dict, topo)

with tf.device("/gpu:0"):
    computation_features = tf.convert_to_tensor(data["features"], dtype=tf.float32)
    device_features = tf.convert_to_tensor(topo["features"], dtype=tf.float32)

    model = GAT(computation_features.shape[1], device_features.shape[1])
    model.set_graphs(data["graph"], topo["graph"])
    print(data["graph"])
    print(topo["graph"])
    print(computation_features.shape)
    print(device_features.shape)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.005, epsilon=1e-8)

    # initialize graph
    dur = []
    for epoch in range(10):
        if epoch >= 3:
            t0 = time.time()
        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)
            decisions = model([computation_features, device_features], training=True)
            s = tf.reduce_sum(decisions)
            loss_value = s + evaluate(data, topo, decisions.numpy()) # numpy to turn off gradient tracking
            grads = tape.gradient(loss_value, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if epoch >= 3:
            dur.append(time.time() - t0)

        print(np.mean(dur) / 1000)
        print(loss_value)
