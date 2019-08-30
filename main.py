import numpy as np
import tensorflow as tf

from mlp import model_fn
from tge import distributify

def show(graph):
    writer = tf.summary.FileWriter('.')
    writer.add_graph(graph)
    writer.flush()

opt = model_fn()
new_opt = distributify(opt)
show(new_opt.graph)

with new_opt.graph.as_default(), tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(new_opt, { x: np.random.uniform(size=(10, 1024)), y: np.random.uniform(size=(10, 10)) })


