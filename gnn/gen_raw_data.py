def vgg(bsize=None):
    from tensorflow.contrib.slim.nets import vgg
    x = tf.placeholder(tf.float32, shape=(bsize, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(bsize, 1000))
    output, _ = vgg.vgg_19(x, 1000)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
    return optimizer

def resnet(bsize=None):
    from tensorflow.contrib.slim.nets import resnet_v2
    x = tf.placeholder(tf.float32, shape=(bsize, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(bsize, 1000))
    output, _ = resnet_v2.resnet_v2_101(x, 1000)
    output = tf.contrib.slim.flatten(output)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
    return optimizer

def mlp(bsize=None):
    x = tf.placeholder(tf.float32, shape=(bsize, 1024))
    y = tf.placeholder(tf.float32, shape=(bsize, 10,))
    hidden = tf.contrib.slim.fully_connected(x, 256, activation_fn=tf.nn.softmax)
    output = tf.contrib.slim.fully_connected(hidden, 10, activation_fn=tf.nn.softmax)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
    return optimizer

def lenet(bsize=None):
    slim = tf.contrib.slim
    x = tf.placeholder(tf.float32, shape=(bsize, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(bsize, 1000))
    net = slim.conv2d(x, 32, [5, 5])
    net = slim.max_pool2d(net, [2, 2], 2)
    net = slim.conv2d(net, 64, [5, 5])
    net = slim.max_pool2d(net, [2, 2], 2)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.sigmoid)
    net = slim.fully_connected(net, 1000, activation_fn=None)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=net)
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(tf.reduce_sum(loss))
    return optimizer

def inception():
    from tensorflow.contrib.slim.nets import inception
    x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(None, 1000))
    output, _ = inception.inception_v3(x, 1000)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
    return optimizer

import tensorflow as tf
import pickle
from profiler import Profiler

BATCHSIZE=120

devices = (
    "/GPU:0",
    "/GPU:1"
)

import sys
model_fn = eval(sys.argv[1])

prof_dict = {}
for nrep in (1, 2, 4, 6, 8):
    tf.reset_default_graph()
    opt = model_fn()
    init = tf.global_variables_initializer()
    gdef = tf.get_default_graph().as_graph_def(add_shapes=True)
    p = Profiler(gdef, BATCHSIZE // nrep)
    for node in gdef.node:
        prof_dict[(node.name, nrep)] = [ p.profile(node.name, device) for device in devices ]
tf.reset_default_graph()
opt = model_fn()
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def(add_shapes=True)
with open("{}.pickle".format(model_fn.__name__), 'wb') as f:
    pickle.dump((gdef, prof_dict), f)
