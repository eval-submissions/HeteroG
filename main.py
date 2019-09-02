import numpy as np
import tensorflow as tf

from mlp import model_fn

def show(graph):
    writer = tf.summary.FileWriter('.')
    writer.add_graph(graph)
    writer.flush()

opt = model_fn()
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def()
bytes = gdef.SerializeToString()
open("g.pb", "wb").write(bytes)

import os
os.system("cargo run")

bytes = open("gout.pb", "rb").read()
g = tf.Graph().as_graph_def()
g.ParseFromString(bytes)
tf.import_graph_def(g)
graph = tf.get_default_graph()

x = graph.get_tensor_by_name("import/Placeholder:0")
y = graph.get_tensor_by_name("import/Placeholder_1:0")
opt = graph.get_operation_by_name("import/GradientDescent")
init = graph.get_operation_by_name("import/init")

show(opt.graph)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)
sess.run(opt, { x: np.random.uniform(size=(10, 1024)), y: np.random.uniform(size=(10, 10)) })


