def model_fn(bsize=None):
    from tensorflow.contrib.slim.nets import vgg
    x = tf.placeholder(tf.float32, shape=(bsize, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(bsize, 1000))
    output, _ = vgg.vgg_19(x, 1000)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
    return optimizer

import time
import numpy as np
import tensorflow as tf
import google.protobuf.text_format as pbtf
from tensorflow.python.client import timeline

from utils import write_tensorboard, setup_workers

opt = model_fn(None)
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def(add_shapes=True)

devices = (
    "/job:tge/replica:0/task:0/device:GPU:0",
    "/job:tge/replica:0/task:0/device:GPU:1"
)

import tge

options = [[0, 1], [1, 0], [0, 2], [2, 0], [1, 1]]
strategy = { node.name: [np.random.randint(0, 2)] + options[np.random.randint(0, len(options))] for node in gdef.node }

g = (tge.TGE(gdef, devices)
    .custom(strategy)
    .compile()
    .get_result()
)

tf.reset_default_graph()
tf.import_graph_def(g)
graph = tf.get_default_graph()

x = graph.get_tensor_by_name("import/Placeholder:0")
y = graph.get_tensor_by_name("import/Placeholder_1:0")
opt = graph.get_operation_by_name("import/GradientDescent")
init = graph.get_operation_by_name("import/init")


data = { x: np.random.uniform(size=(64, 224, 224, 3)), y: np.random.uniform(size=(64, 1000)) }
config = tf.ConfigProto(allow_soft_placement=True)#log_device_placement=True)

sess = tf.Session(server.target, config=config)
sess.run(init)
sess.run(opt, data)

for i in range(3):
    tic = time.perf_counter()
    sess.run(opt, data)
    toc = time.perf_counter()
    print("actual {}: {}".format(i, toc - tic))

tf.reset_default_graph()
opt = model_fn(64)
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def(add_shapes=True)

g = (tge.TGE(gdef, devices)
    .custom(strategy)
    .set_bandwidth(intra=100000, inter=1000)
    .evaluate({ node.name: [np.random.randint(0, 1000)] * len(devices) for node in gdef.node })
)
print("simulated: {}".format(g[0]))
