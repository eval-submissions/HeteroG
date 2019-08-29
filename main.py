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

with new_opt.graph.as_default():
    sess = tf.Session()
    sess.run(new_opt)

