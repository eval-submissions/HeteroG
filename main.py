import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=(None, 1024))
y = tf.placeholder(tf.float32, shape=(None, 10,))
hidden = tf.contrib.slim.fully_connected(x, 256, activation_fn=tf.nn.softmax)
output = tf.contrib.slim.fully_connected(hidden, 10, activation_fn=tf.nn.softmax)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))

dag = optimizer.graph.as_graph_def().node

graph = tf.get_default_graph()


for node in graph.as_graph_def().node:
    node.device = '/device:CPU:0'


with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(optimizer, { x: np.random.uniform(size=(10, 1024)), y: np.random.uniform(size=(10, 10)) })

# design:
# 1. for the model part, we need only the optimizer_op

