def model_fn():
    from tensorflow.contrib.slim.nets import vgg
    x = tf.placeholder(tf.float32, shape=(64, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(64, 1000))
    output, _ = vgg.vgg_19(x, 1000)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
    return optimizer

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import numpy as np
import tensorflow as tf

opt = model_fn()
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def(add_shapes=True)

devices = (
    "/job:tge/replica:0/task:0/device:GPU:0",
    "/job:tge/replica:0/task:0/device:GPU:1",
    "/job:tge/replica:0/task:1/device:GPU:0",
    "/job:tge/replica:0/task:1/device:GPU:1"
)

import tge

# with open("../../xiaodong/tge/GAT/data/graph/docs.txt", "r") as f:
#     records = (x.strip().split(" ") for x in f.readlines())
#     prof = {items[0]: [int(float(x)) for x in items[1:]] for items in records}

g = (tge.TGE(gdef, devices)
    .custom({ node.name: [0, 2, 1, 0, 1] for node in gdef.node })
    .evaluate({ node.name: [200]*len(devices) for node in gdef.node }, "trace.json")
)
print(g)
