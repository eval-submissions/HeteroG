def model_fn():
    x = tf.placeholder(tf.float32, shape=(64, 1024))
    y = tf.placeholder(tf.float32, shape=(64, 10,))
    hidden = tf.layers.dense(x, 256, activation=tf.nn.softmax)
    output = tf.layers.dense(hidden, 10, activation=tf.nn.softmax)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
    return optimizer

import numpy as np
import tensorflow.compat.v1 as tf
import google.protobuf.text_format as pbtf

tf.disable_eager_execution()

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

noop = ('Placeholder', 'Const', 'Identity', 'NoOp', 'ReadVariableOp', 'VarHandleOp', 'Shape')

g = (tge.TGE(gdef, devices)
    .custom({ node.name: (0, 1, 1, 1, 1) for node in gdef.node })
    .set_bandwidth(10000, 100)
    .evaluate({ node.name: [0 if node.op in noop else 1000] * len(devices) for node in gdef.node })
)
print(g)
