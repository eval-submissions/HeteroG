def model_fn():
    from tensorflow.contrib.slim.nets import vgg
    x = tf.placeholder(tf.float32, shape=(None, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(None, 1000))
    output, _ = vgg.vgg_19(x)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
    return optimizer

import time
import subprocess as sb
import numpy as np
import tensorflow as tf
import google.protobuf.text_format as pbtf

from utils import write_tensorboard, restart_workers

opt = model_fn()
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def(add_shapes=True)
bytes = gdef.SerializeToString()

devices = (
    "/job:tge/replica:0/task:0/device:GPU:0",
    "/job:tge/replica:0/task:0/device:GPU:1",
    "/job:tge/replica:0/task:1/device:GPU:0",
    "/job:tge/replica:0/task:1/device:GPU:1"
)

tic1 = time.perf_counter()
p = sb.Popen(["./tge", *devices], stdin=sb.PIPE, stdout=sb.PIPE)
bytes, _ = p.communicate(bytes)
toc1 = time.perf_counter()

g = tf.Graph().as_graph_def()
g.ParseFromString(bytes)
tf.import_graph_def(g)
graph = tf.get_default_graph()

x = graph.get_tensor_by_name("import/Placeholder:0")
y = graph.get_tensor_by_name("import/Placeholder_1:0")
opt = graph.get_operation_by_name("import/GradientDescent")
init = graph.get_operation_by_name("import/init")

data = { x: np.random.uniform(size=(48, 224, 224, 3)), y: np.random.uniform(size=(48, 1000)) }

# dag = tf.graph_util.extract_sub_graph(dag, [op.name, init.name])

write_tensorboard(opt.graph)

workers = ["net-g11:3901", "net-g10:3901"]
restart_workers(workers)
server = tf.distribute.Server(tf.train.ClusterSpec({
    "tge": workers
}), job_name='tge', task_index=0)

sess = tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True))
sess.run(init)
sess.run(opt, data) # heat up

profiler = tf.profiler.Profiler(graph)
for i in range(4):
    run_meta = tf.compat.v1.RunMetadata()
    sess.run(opt, data,
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

tic2 = time.perf_counter()
for i in range(10):
    sess.run(opt, data)

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

