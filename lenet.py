def model_fn():
    slim = tf.contrib.slim
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, 10))
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

import time
import subprocess as sb
import numpy as np
import tensorflow as tf

from utils import write_tensorboard

opt = model_fn()
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def() # add_shapes=True? then we must keep tracking shapes ourselves
bytes = gdef.SerializeToString()

devices = (
    # "/job:tge/replica:0/task:0/device:GPU:0",
    # "/job:tge/replica:0/task:0/device:GPU:1",
    "/job:tge/replica:0/task:1/device:GPU:0",
    "/job:tge/replica:0/task:1/device:GPU:1"
)

tic1 = time.perf_counter()
p = sb.Popen(["./tge", *devices], stdin=sb.PIPE, stdout=sb.PIPE)
bytes, _ = p.communicate(bytes)
toc1 = time.perf_counter()

g = tf.Graph().as_graph_def()
g.ParseFromString(bytes)
tf.reset_default_graph()
tf.import_graph_def(g)
graph = tf.get_default_graph()

x = graph.get_tensor_by_name("import/Placeholder:0")
y = graph.get_tensor_by_name("import/Placeholder_1:0")
opt = graph.get_operation_by_name("import/GradientDescent")
init = graph.get_operation_by_name("import/init")
# currently a hack. Later we will add an API for user to get tensor references back
acc = 10 * (graph.get_tensor_by_name("import/Mean/replica_0:0") + graph.get_tensor_by_name("import/Mean/replica_1:0")) / 2

write_tensorboard(opt.graph)

server = tf.distribute.Server(tf.train.ClusterSpec({
    "tge": ["net-g10:3901", "net-g11:3901"]
}), job_name='tge', task_index=1)

sess = tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True))
sess.run(init)

def onehot(x):
    max = x.max() + 1
    return np.eye(max)[x]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255
y_train = onehot(y_train.reshape(-1))
y_test = onehot(y_test.reshape(-1))
batch_size = 50

tpi = []
tic2 = time.perf_counter()
for batch_id in range(2000):
    i = batch_id % 1000
    tic = time.perf_counter()
    sess.run(opt, { x: x_train[batch_size*i:batch_size*(i+1)], y: y_train[batch_size*i:batch_size*(i+1)] })
    toc = time.perf_counter()
    tpi.append(toc - tic)
    if i % 100 == 0:
        a = sess.run(acc, { x: x_test, y: y_test })
        print("acc: ", a)
        print("tpi: ", sum(tpi[-50:-1]) / 50)

toc2 = time.perf_counter()

import sys
sys.stderr.write("""

================

planning time: {plan_time}s
trainning time: {train_time}s
time per iter: {tpi}s

""".format(
    plan_time = toc1 - tic1,
    train_time = toc2 - tic2,
    tpi = sum(tpi) / len(tpi)
))

