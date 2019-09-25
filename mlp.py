def model_fn():
    x = tf.placeholder(tf.float32, shape=(None, 1024))
    y = tf.placeholder(tf.float32, shape=(None, 10,))
    hidden = tf.contrib.slim.fully_connected(x, 256, activation_fn=tf.nn.softmax)
    output = tf.contrib.slim.fully_connected(hidden, 10, activation_fn=tf.nn.softmax)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
    return optimizer

import time
import numpy as np
import tensorflow as tf

from utils import write_tensorboard, setup_workers

opt = model_fn()
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def(add_shapes=True)
bytes = gdef.SerializeToString()

devices = (
    "/job:tge/replica:0/task:0/device:GPU:0",
    # "/job:tge/replica:0/task:0/device:GPU:1",
    "/job:tge/replica:0/task:1/device:GPU:0",
    # "/job:tge/replica:0/task:1/device:GPU:1"
)

import tge
g = (tge.TGE()
    .set_graph_def(gdef)
    .set_devices(devices)
    .data_parallel('ps0')
    .compile()
    .get_graph_def()
)

tf.reset_default_graph()
tf.import_graph_def(g)
graph = tf.get_default_graph()

x = graph.get_tensor_by_name("import/Placeholder:0")
y = graph.get_tensor_by_name("import/Placeholder_1:0")
opt = graph.get_operation_by_name("import/GradientDescent")
init = graph.get_operation_by_name("import/init")

write_tensorboard(opt.graph)

workers = ["10.28.1.26:3901", "10.28.1.25:3901"]
server = setup_workers(workers)

sess = tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True))
sess.run(init)

tic2 = time.perf_counter()
for i in range(100):
    sess.run(opt, { x: np.random.uniform(size=(120, 1024)), y: np.random.uniform(size=(120, 10)) })
toc2 = time.perf_counter()

import sys
sys.stderr.write("""

================

planning time: {plan_time}s
trainning time: {train_time}s

""".format(
    plan_time = toc1 - tic1,
    train_time = toc2 - tic2
))

