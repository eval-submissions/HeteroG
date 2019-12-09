import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import tensorflow as tf

def vgg_19():
    from tensorflow.contrib.slim.nets import vgg
    x = tf.placeholder(tf.float32, shape=(64, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(64, 1000))
    output, _ = vgg.vgg_19(x, 1000)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
    return optimizer

def mlp():
    x = tf.placeholder(tf.float32, shape=(64, 1024))
    y = tf.placeholder(tf.float32, shape=(64, 10,))
    hidden = tf.contrib.slim.fully_connected(x, 256, activation_fn=tf.nn.softmax)
    output = tf.contrib.slim.fully_connected(hidden, 10, activation_fn=tf.nn.softmax)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
    return optimizer

def lenet():
    slim = tf.contrib.slim
    x = tf.placeholder(tf.float32, shape=(64, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(64, 10))
    net = slim.conv2d(x, 32, [5, 5])
    net = slim.max_pool2d(net, [2, 2], 2)
    net = slim.conv2d(net, 64, [5, 5])
    net = slim.max_pool2d(net, [2, 2], 2)
    net = slim.flatten(net)
    net = slim.fully_connected(net, 1024, activation_fn=tf.nn.sigmoid)
    net = slim.fully_connected(net, 10, activation_fn=None)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=net)
    acc = tf.reduce_mean(tf.nn.softmax(net) * y)
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(tf.reduce_sum(loss))
    return optimizer

for f in ['vgg_19', 'mlp', 'lenet']:
    tf.reset_default_graph()
    opt = eval(f)()
    init = tf.global_variables_initializer()
    gdef = tf.get_default_graph().as_graph_def(add_shapes=True)
    with open('{}.pb'.format(f), 'wb') as fout:
        fout.write(gdef.SerializeToString())
