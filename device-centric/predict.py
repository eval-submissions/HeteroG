import numpy as np
import time
import tensorflow as tf

from model import Model
from data import gen_topo, gen_data
from utils import save, load
from environment import sample

with tf.device("/gpu:1"):
    records = load("records")
    model = Model(4, 2, 2, 3, 8, records[0]["op_table"])
    model.load_weights('weights')

    gdef, prof_data = load("vgg.pickle")

    for bandwidth in (2 ** n for n in range(8, 20)):
        topo = gen_topo([
            ("/job:worker/replica:0/task:0/device:GPU:0", 1, 12<<30),
            ("/job:worker/replica:0/task:0/device:GPU:1", 1, 12<<30),
            ("/job:worker/replica:0/task:0/device:GPU:2", 1, 12<<30),
            ("/job:worker/replica:0/task:0/device:GPU:3", 1, 12<<30),
            ("/job:worker/replica:0/task:1/device:GPU:0", 1, 12<<30),
        ], intra=bandwidth, inter=100)
        record = gen_data(gdef, prof_data, topo, records[0]["op_table"])

        cnfeats = tf.convert_to_tensor(record["cnfeats"], dtype=tf.float32)
        cefeats = tf.convert_to_tensor(record["cefeats"], dtype=tf.float32)
        cntypes = tf.convert_to_tensor(record["cntypes"], dtype=tf.float32)
        tnfeats = tf.convert_to_tensor(record["tnfeats"], dtype=tf.float32)
        tefeats = tf.convert_to_tensor(record["tefeats"], dtype=tf.float32)
        model.set_graphs(record["cgraph"], record["tgraph"])
        model.set_groups(record["cgroups"], record["tgroups"])

        logp = model([cnfeats, cefeats, cntypes, tnfeats, tefeats])
        p = np.argmax(logp.numpy()[:, 1:], axis=1)
        count = {}
        for i in range(p.shape[0]):
            d = (1 if logp.numpy()[i, 0] > .5 else 0), p[i]
            count[d] = count.get(d, 0) + 1
        for d, c in sorted(list(count.items()), key=lambda x: -x[1]):
            print("{},{}/{}:".format(int(d[0]), d[1], tuple(record["tgroups"][d[1]])), c)

        count = {}
        for i in range(p.shape[0]):
            d = p[i]
            count[d] = count.get(d, 0) + 1
        print("\n=== bandwidth={} ===".format(bandwidth))
        for k, v in count.items():
            print(k, v)
