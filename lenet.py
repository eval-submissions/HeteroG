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

import subprocess as sb
import numpy as np
import tensorflow as tf
import google.protobuf.text_format as pbtf

from utils import write_tensorboard, setup_workers

opt = model_fn()
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def() # add_shapes=True? then we must keep tracking shapes ourselves
bytes = gdef.SerializeToString()

devices = (
    "/job:tge/replica:0/task:0/device:GPU:0",
    "/job:tge/replica:0/task:0/device:GPU:1",
    "/job:tge/replica:0/task:1/device:GPU:0",
    "/job:tge/replica:0/task:1/device:GPU:1"
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
# currently a hack. Later we will add an API for user to get tensor references back
acc = 10 * (graph.get_tensor_by_name("import/Mean/replica_0:0") + graph.get_tensor_by_name("import/Mean/replica_1:0")) / 2

write_tensorboard(opt.graph)

workers = ["10.28.1.26:3901", "10.28.1.25:3901"]
server = setup_workers(workers)

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
batch_size = 40

profiler = tf.profiler.Profiler(graph)
for i in range(4):
    run_meta = tf.compat.v1.RunMetadata()
    sess.run(opt,
        { x: x_train[batch_size*i:batch_size*(i+1)], y: y_train[batch_size*i:batch_size*(i+1)] },
        options=tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        run_metadata=run_meta
    )
    profiler.add_step(i, run_meta)

    r = profiler.profile_operations(options=tf.profiler.ProfileOptionBuilder.time_and_memory())
    with open("p_{}".format(i), "w") as fo:
        fo.write(pbtf.MessageToString(r))

profiler.profile_graph(options=
    tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.time_and_memory())
        .with_timeline_output("t").build())

tpi = []
for batch_id in range(2000):
    i = batch_id % 1250
    run_meta = tf.compat.v1.RunMetadata()

    tic = time.perf_counter()
    sess.run(opt,
        { x: x_train[batch_size*i:batch_size*(i+1)], y: y_train[batch_size*i:batch_size*(i+1)] },
        options=tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
        run_metadata=run_meta
    )
    toc = time.perf_counter()
    tpi.push(toc - tic)

    if i % 50 == 0:
        a = sess.run(acc, { x: x_test, y: y_test })
        print("acc: ", a)
        print("tpi: ", sum(tpi[-50:-1]) / 50)
