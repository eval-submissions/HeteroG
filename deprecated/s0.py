import numpy as np
import tensorflow as tf
import pickle

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

g = pickle.load(open("model.pickle", "rb"))
tf.import_graph_def(g)

server = tf.distribute.Server(tf.train.ClusterSpec({
    "tge": ["127.0.0.1:3901", "127.0.0.1:3902"]
}), job_name='tge', task_index=0, protocol="grpc")

graph = tf.get_default_graph()

x = graph.get_tensor_by_name("import/Placeholder/replica_0:0")
y = graph.get_tensor_by_name("import/Placeholder_1/replica_0:0")
opt = graph.get_operation_by_name("import/GradientDescent/replica_0")
init = graph.get_operation_by_name("import/init/replica_0")

data = { x: np.random.uniform(size=(64, 224, 224, 3)), y: np.random.uniform(size=(64, 1000)) }

sess = tf.Session(server.target)
sess.run(init)
sess.run(opt, data)
# sess.run(opt)

run_meta = tf.compat.v1.RunMetadata()
run_opt = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
sess.run(opt, data, options=run_opt, run_metadata=run_meta)

print("done")
