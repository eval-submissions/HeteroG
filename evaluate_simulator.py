def model_fn(bsize=None):
    from tensorflow.contrib.slim.nets import vgg
    x = tf.placeholder(tf.float32, shape=(bsize, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(bsize, 1000))
    output, _ = vgg.vgg_19(x, 1000)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
    return optimizer

# def model_fn(bsize=None):
#     from tensorflow.contrib.slim.nets import resnet_v2
#     x = tf.placeholder(tf.float32, shape=(bsize, 224, 224, 3))
#     y = tf.placeholder(tf.float32, shape=(bsize, 1000))
#     output, _ = resnet_v2.resnet_v2_101(x, 1000)
#     output = tf.contrib.slim.flatten(output)
#     loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
#     optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
#     return optimizer

# def model_fn(bsize):
#     x = tf.placeholder(tf.float32, shape=(bsize, 1024))
#     y = tf.placeholder(tf.float32, shape=(bsize, 10,))
#     hidden = tf.contrib.slim.fully_connected(x, 256, activation_fn=tf.nn.softmax)
#     output = tf.contrib.slim.fully_connected(hidden, 10, activation_fn=tf.nn.softmax)
#     loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
#     optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
#     return optimizer

# def model_fn(bsize):
#     slim = tf.contrib.slim
#     x = tf.placeholder(tf.float32, shape=(bsize, 32, 32, 3))
#     y = tf.placeholder(tf.float32, shape=(bsize, 1000))
#     net = slim.conv2d(x, 32, [5, 5])
#     net = slim.max_pool2d(net, [2, 2], 2)
#     net = slim.conv2d(net, 64, [5, 5])
#     net = slim.max_pool2d(net, [2, 2], 2)
#     net = slim.flatten(net)
#     net = slim.fully_connected(net, 1024, activation_fn=tf.nn.sigmoid)
#     net = slim.fully_connected(net, 1000, activation_fn=None)
#     loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=net)
#     acc = tf.reduce_mean(tf.nn.softmax(net) * y)
#     optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(tf.reduce_sum(loss))
#     return optimizer


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
    "/GPU:0",
    "/GPU:1"
)

import tge

# options = [[0, 1], [1, 0], [0, 2], [2, 0], [1, 1]]
# strategy = { node.name: [np.random.randint(0, 2)] + options[np.random.randint(0, len(options))] for node in gdef.node }

strategy = { node.name: [0, 3, 1] for node in gdef.node }

g = (tge.TGE(gdef, devices)
    .custom(strategy)
    .compile()
    .get_result()
)

tf.reset_default_graph()
tf.import_graph_def(g)
graph = tf.get_default_graph()

x = graph.get_tensor_by_name("import/Placeholder/replica_0:0")
y = graph.get_tensor_by_name("import/Placeholder_1/replica_0:0")
opt = graph.get_operation_by_name("import/GradientDescent/replica_0")
init = graph.get_operation_by_name("import/init/replica_0")

data = { x: np.random.uniform(size=(64, 224, 224, 3)), y: np.random.uniform(size=(64, 1000)) }
config = tf.ConfigProto(allow_soft_placement=True)#log_device_placement=True)

sess = tf.Session(None, config=config)
sess.run(init)
sess.run(opt, data)

# op = graph.get_tensor_by_name("import/vgg_19/fc6/Conv2D/replica_0:0")
# print(sess.run(op, data).shape)

run_meta = tf.compat.v1.RunMetadata()
run_opt = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
sess.run(opt, data, options=run_opt, run_metadata=run_meta)

with open("meta.pb", "w") as fo:
    fo.write(pbtf.MessageToString(run_meta))

tl = timeline.Timeline(run_meta.step_stats)
with open("timeline.json", "w") as fo:
    fo.write(tl.generate_chrome_trace_format())

tic = time.perf_counter()
sess.run(opt, data)
toc = time.perf_counter()

tf.reset_default_graph()
opt = model_fn(64)
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def(add_shapes=True)

with open("model.pb", "w") as fo:
    fo.write(pbtf.MessageToString(gdef))

from profiler import Profiler

p = Profiler(gdef)
prof_dict = { node.name: [ p.profile(node.name, device) for device in devices ] for node in gdef.node }

g = (tge.TGE(gdef, devices)
    .custom(strategy)
    .set_bandwidth(intra=6000, inter=6000)
    .evaluate(prof_dict, "simulated.json")
)

print("actual: {}".format(toc - tic))
print("simulated: {}".format(g[0]))

with open("result", "a") as fo:
    fo.write("{} {}\n".format(toc - tic, g[0]))
