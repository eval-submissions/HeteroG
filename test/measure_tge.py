# def model_fn(bsize=None):
#     from tensorflow.contrib.slim.nets import vgg
#     x = tf.placeholder(tf.float32, shape=(bsize, 224, 224, 3))
#     y = tf.placeholder(tf.float32, shape=(bsize, 1000))
#     output, _ = vgg.vgg_19(x, 1000)
#     loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
#     optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
#     return optimizer

def model_fn(bsize=None):
    from tensorflow.contrib.slim.nets import resnet_v2
    x = tf.placeholder(tf.float32, shape=(bsize, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(bsize, 1000))
    output, _ = resnet_v2.resnet_v2_101(x, 1000)
    output = tf.contrib.slim.flatten(output)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
    return optimizer

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
import tensorflow as tf
import pickle

BATCHSIZE=48

devices = (
    "/job:worker/replica:0/task:0/device:GPU:0",
    "/job:worker/replica:0/task:0/device:GPU:1",
    "/job:worker/replica:0/task:1/device:GPU:0",
    "/job:worker/replica:0/task:1/device:GPU:1",
    "/job:worker/replica:0/task:2/device:GPU:0",
    "/job:worker/replica:0/task:2/device:GPU:1",
    "/job:worker/replica:0/task:3/device:GPU:0",
    "/job:worker/replica:0/task:3/device:GPU:1",
    "/job:worker/replica:0/task:4/device:GPU:0",
    "/job:worker/replica:0/task:4/device:GPU:1",
    "/job:worker/replica:0/task:5/device:GPU:0",
    "/job:worker/replica:0/task:5/device:GPU:1",
)

from tge import TGE
from profiler import Profiler

try:
    prof_dict = pickle.load(open("measure_tge.pickle", 'rb'))
except:
    prof_dict = {}
    for nrep in (1, 2, 3, 4, 6, 8, 12):
        tf.reset_default_graph()
        opt = model_fn(BATCHSIZE // nrep)
        init = tf.global_variables_initializer()
        gdef = tf.get_default_graph().as_graph_def(add_shapes=True)
        p = Profiler(gdef)
        for node in gdef.node:
            prof_dict[(node.name, nrep)] = [ p.profile(node.name, '/GPU:0') for device in devices ]
    pickle.dump(prof_dict, open("measure_tge.pickle", 'wb'))

tic = time.time()
opt = model_fn(BATCHSIZE)
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def(add_shapes=True)
print("reference: ", time.time() - tic)

strategy = { node.name: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] for node in gdef.node }

tic = time.time()
tge = TGE(gdef, devices)
print("read: ", time.time() - tic)
tic = time.time()
tge.custom(strategy)
tge.replace_placeholder(BATCHSIZE)
tge.set_bandwidth(intra=2810, inter=2810)
print("prepare: ", time.time() - tic)
tic = time.time()
tge.compile()
print("compile: ", time.time() - tic)
tic = time.time()
tge.evaluate(prof_dict)
print("simulate: ", time.time() - tic)

