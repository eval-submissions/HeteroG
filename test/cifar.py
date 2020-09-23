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
    optimizer = tf.train.GradientDescentOptimizer(0.002).minimize(tf.reduce_sum(loss))
    return optimizer

import numpy as np
import tensorflow as tf
import google.protobuf.text_format as pbtf
from tensorflow.python.client import timeline
from tensorflow.distribute.cluster_resolver import TFConfigClusterResolver
from utils import info

import os
os.environ["TF_CONFIG"] = '{ "cluster": { "worker": ["10.28.1.24:3806", "10.28.1.16:3901"] }, "task": {"type": "worker", "index": 0} }'

def setup_workers(workers, protocol="grpc"):
    import urllib.request
    import time

    param = '/'.join(server.replace(':', '%3A') for server in workers)
    for task_id, server in enumerate(workers):
        if task_id == 0: continue
        url = "http://{}:3905/{}/restart/{}/{}/{}".format(server.split(':')[0], int(time.time()) + 10, protocol, task_id, param)
        assert urllib.request.urlopen(url).read() == b'ok'
    time.sleep(1)

setup_workers(["10.28.1.24:3806", "10.28.1.16:3901"])

BATCHSIZE=40

devices = (
    "/job:worker/replica:0/task:0/device:GPU:0",
    "/job:worker/replica:0/task:0/device:GPU:1",
    "/job:worker/replica:0/task:1/device:GPU:0",
    "/job:worker/replica:0/task:1/device:GPU:1"
)
resolver = TFConfigClusterResolver()
cluster = resolver.cluster_spec()
dist = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.NCCL)
config = dist.update_config_proto(tf.ConfigProto())
config.ClearField("device_filters")
server = tf.distribute.Server(cluster, job_name='worker', task_index=0, protocol="grpc", config=config)

opt = model_fn()
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def(add_shapes=True)

with open("model.pb", "w") as fo:
    fo.write(pbtf.MessageToString(gdef))

import tge

strategy = { node.name: [1, 1, 1, 1, 1] for node in gdef.node }

g = (tge.TGE(gdef, devices)
    .custom(strategy)
    # .replace_placeholder(BATCHSIZE)
    .use_collective()
    # .verbose()
    .compile()
    .get_result()
)

with open("modified.pb", "w") as fo:
    fo.write(pbtf.MessageToString(g))

tf.reset_default_graph()
resolver = TFConfigClusterResolver()
cluster = resolver.cluster_spec()
dist = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.NCCL)
config = dist.update_config_proto(tf.ConfigProto())
config.ClearField("device_filters")
tf.import_graph_def(g)
graph = tf.get_default_graph()

x_tensor = graph.get_tensor_by_name("import/Placeholder/replica_0:0")
y_tensor = graph.get_tensor_by_name("import/Placeholder_1/replica_0:0")
opt = graph.get_operation_by_name("import/GradientDescent/replica_0")
init = graph.get_operation_by_name("import/init/replica_0")
acc_tensor = 10 * (
    graph.get_tensor_by_name("import/Mean/replica_0:0") +
    graph.get_tensor_by_name("import/Mean/replica_1:0") +
    graph.get_tensor_by_name("import/Mean/replica_2:0") +
    graph.get_tensor_by_name("import/Mean/replica_3:0")) / 4

sess = tf.Session(server.target, config=config)
sess.run(init)

def onehot(x):
    max = x.max() + 1
    return np.eye(max)[x]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255
x_test = x_test / 255
y_train = onehot(y_train.reshape(-1))
y_test = onehot(y_test.reshape(-1))

for batch_id in range(5000):
    i = batch_id % 1250

    sess.run(opt, {
        x_tensor: x_train[BATCHSIZE*i:BATCHSIZE*(i+1)],
        y_tensor: y_train[BATCHSIZE*i:BATCHSIZE*(i+1)]
    })

    if i % 50 == 0:
        a = sess.run(acc_tensor, { x_tensor: x_test, y_tensor: y_test })
        info("acc:", a)
