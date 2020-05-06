import numpy as np
import time
import tensorflow as tf
import pickle

from model import GAT
from data import gen_topo, gen_data

with tf.device("/gpu:1"):
    model = GAT(4, 2)
    model.load_weights('weights')

    gdef, prof_data = pickle.load(open("vgg.pickle", "rb"))

    for bandwidth in (2 ** n for n in range(4, 12)):
        topo = gen_topo([
            ("/job:worker/replica:0/task:0/device:GPU:0", 1),
            ("/job:worker/replica:0/task:0/device:GPU:1", 1),
            ("/job:worker/replica:0/task:0/device:GPU:0", 1.2),
            ("/job:worker/replica:0/task:0/device:GPU:1", 1.2),
            ("/job:worker/replica:0/task:0/device:GPU:0", 1.5),
            ("/job:worker/replica:0/task:0/device:GPU:1", 1.5),
        ], intra=bandwidth)
        record = gen_data(gdef, prof_data, topo)

        computation_features = tf.convert_to_tensor(record["computation_features"], dtype=tf.float32)
        device_features = tf.convert_to_tensor(record["device_features"], dtype=tf.float32)
        model.set_graphs(record["computation_graph"], record["device_graph"])
        logp = model([computation_features, device_features])
        p = np.argmax(logp.numpy(), axis=2)
        count = {}
        for i in range(p.shape[0]):
            d = tuple(p[i, :])
            count[d] = count.get(d, 0) + 1
        avg_num_device = 0
        for k, v in count.items():
            avg_num_device += v * max(1, sum((1 for x in k if x > 0)))
        avg_num_device /= sum(count.values())
        print("\n=== bandwidth={} ===".format(bandwidth))
        print("avg_num_device:", avg_num_device)
        for k, v in count.items():
            print(k, v)
