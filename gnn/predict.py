import numpy as np
import time
import tensorflow as tf

from model import Model
from data import gen_topo, gen_data
from utils import save, load
from environment import sample, kmeans_sample

with tf.device("/gpu:1"):
    model = Model(4, 1, 2, 3)
    model.load_weights('weights')

    gdef, prof_data = load("vgg.pickle")

    for bandwidth in (2 ** n for n in range(8, 20)):
        topo = gen_topo([
            ("/job:worker/replica:0/task:0/device:GPU:0", 1, 6<<30),
            ("/job:worker/replica:0/task:0/device:GPU:1", 1, 6<<30),
            ("/job:worker/replica:0/task:0/device:GPU:2", 1, 6<<30),
            ("/job:worker/replica:0/task:0/device:GPU:3", 1, 6<<30),
            ("/job:worker/replica:0/task:1/device:GPU:0", 1, 6<<30),
        ], intra=bandwidth, inter=10)
        record = gen_data(gdef, prof_data, topo)

        cnfeats = tf.convert_to_tensor(record["cnfeats"], dtype=tf.float32)
        cefeats = tf.convert_to_tensor(record["cefeats"], dtype=tf.float32)
        tnfeats = tf.convert_to_tensor(record["tnfeats"], dtype=tf.float32)
        tefeats = tf.convert_to_tensor(record["tefeats"], dtype=tf.float32)
        model.set_graphs(record["cgraph"], record["tgraph"])
        model.set_groups(record["groups"])

        logp = model([cnfeats, cefeats, tnfeats, tefeats])
        # p = np.argmax(logp.numpy(), axis=2)
        # _, p = sample(logp.numpy())
        _, p = kmeans_sample(logp.numpy())
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
