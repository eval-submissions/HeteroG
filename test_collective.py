def model_fn(bsize=None):
    from tensorflow.contrib.slim.nets import vgg
    x = tf.placeholder(tf.float32, shape=(bsize, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(bsize, 10))
    output, _ = vgg.vgg_19(x, 10)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
    return optimizer


import sys
import numpy as np
import tensorflow as tf
import google.protobuf.text_format as pbtf
from tensorflow.python.client import timeline
from tensorflow.distribute.cluster_resolver import TFConfigClusterResolver

tf.logging.set_verbosity('DEBUG')

devices = (
    "/job:worker/replica:0/task:0/device:GPU:0",
    "/job:worker/replica:0/task:1/device:GPU:0"
)
# config = tf.ConfigProto(allow_soft_placement=True, experimental={"collective_group_leader": "/job:worker/replica:0/task:0", "collective_nccl":True})
resolver = TFConfigClusterResolver()
cluster = resolver.cluster_spec()
dist = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.NCCL)
config = dist.update_config_proto(tf.ConfigProto())
config.ClearField("device_filters")
server = tf.distribute.Server(cluster, job_name='worker', task_index=0, protocol="grpc", config=config)
# devices = ("GPU:0", "GPU:1")

opt = model_fn()
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def(add_shapes=True)

import tge

strategy = { node.name: [1, 2, 2] for node in gdef.node }

g = (tge.TGE(gdef, devices)
    .custom(strategy)
    # .use_nccl()
    .replace_placeholder(48)
    .use_collective()
    # .verbose()
    .compile()
    .get_result()
)

tf.reset_default_graph()
tf.import_graph_def(g)
graph = tf.get_default_graph()

# x = graph.get_tensor_by_name("import/Placeholder/replica_0:0")
# y = graph.get_tensor_by_name("import/Placeholder_1/replica_0:0")
opt = graph.get_operation_by_name("import/GradientDescent/replica_0")
init = graph.get_operation_by_name("import/init/replica_0")

# data = { x: np.random.uniform(size=(24, 224, 224, 3)), y: np.random.uniform(size=(24, 10)) }

sess = tf.Session(server.target, config=config)
# sess = tf.Session(config=config)
sess.run(init)
sess.run(opt)

# run_meta = tf.compat.v1.RunMetadata()
# run_opt = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# sess.run(opt, data, options=run_opt, run_metadata=run_meta)
#
# with open("meta.pb", "w") as fo:
#     fo.write(pbtf.MessageToString(run_meta))
#
# tl = timeline.Timeline(run_meta.step_stats)
# with open("timeline.json", "w") as fo:
#     fo.write(tl.generate_chrome_trace_format())

print("done")

# CUDA_VISIBLE_DEVICES=1 python
# import tensorflow as tf
# tf.distribute.Server(tf.train.ClusterSpec({
#     "worker": ["127.0.0.1:3901", "127.0.0.1:3902"]
# }), job_name='worker', task_index=1, protocol="grpc").join()
