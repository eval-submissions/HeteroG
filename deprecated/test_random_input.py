def model_fn(bsize=None):
    from tensorflow.contrib.slim.nets import vgg
    x = tf.placeholder(tf.float32, shape=(bsize, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(bsize, 10))
    output, _ = vgg.vgg_19(x, 10)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
    return optimizer

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import numpy as np
import tensorflow as tf
import google.protobuf.text_format as pbtf
from tensorflow.python.client import timeline

opt = model_fn()
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def(add_shapes=True)

# devices = (
#     "/job:tge/replica:0/task:0/device:GPU:0",
#     "/job:tge/replica:0/task:1/device:GPU:0"
# )
# server = tf.distribute.Server(tf.train.ClusterSpec({
#     "tge": ["127.0.0.1:3901", "127.0.0.1:3902"]
# }), job_name='tge', task_index=0, protocol="grpc")
devices = ("GPU:0", "GPU:1")

import tge

options = [[1, 1], [2, 2]]
strategy = { node.name: [1] + options[np.random.randint(0, len(options))] for node in gdef.node }

g = (tge.TGE(gdef, devices)
    .custom(strategy)
    # .replace_placeholder(64)
    .use_nccl()
    # .verbose()
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

data = { x: np.random.uniform(size=(24, 224, 224, 3)), y: np.random.uniform(size=(24, 10)) }

# sess = tf.Session(server.target)
sess = tf.Session()
sess.run(init)
sess.run(opt, data)
# sess.run(opt)

run_meta = tf.compat.v1.RunMetadata()
run_opt = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
sess.run(opt, data, options=run_opt, run_metadata=run_meta)

with open("meta.pb", "w") as fo:
    fo.write(pbtf.MessageToString(run_meta))

tl = timeline.Timeline(run_meta.step_stats)
with open("timeline.json", "w") as fo:
    fo.write(tl.generate_chrome_trace_format())

print("done")

# CUDA_VISIBLE_DEVICES=1 python
# import tensorflow as tf
# tf.distribute.Server(tf.train.ClusterSpec({
#     "tge": ["127.0.0.1:3901", "127.0.0.1:3902"]
# }), job_name='tge', task_index=1, protocol="grpc").join()
