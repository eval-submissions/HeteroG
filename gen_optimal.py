import numpy as np
import tensorflow as tf

def model_fn():
    x = tf.placeholder(tf.float32, shape=(256, 102400))
    y = tf.placeholder(tf.float32, shape=(256, 1024))
    w = tf.Variable(tf.random_normal([102400, 1024]))
    o = tf.matmul(x, w)
    l = tf.reduce_sum(y - o)
    go = l * o
    gw = tf.matmul(tf.transpose(x), go)
    opt = tf.train.GradientDescentOptimizer(0.2)
    opt = opt.apply_gradients([(gw, w)])
    return opt

opt = model_fn()
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def(add_shapes=True)
for node in gdef.node:
    n = [x for x in node.input if not x.endswith("group_deps")]
    del node.input[:]
    node.input.extend(n)

gdef = tf.graph_util.remove_training_nodes(gdef)
n = [node for node in gdef.node if not node.name.endswith("group_deps")]
del gdef.node[:]
gdef.node.extend(n)

from profiler import Profiler
p = Profiler(gdef)

with open("mlp.data", "w") as fout:
    for node in gdef.node:
        t = p.profile(node.name, "/gpu:0")
        print(node.name, t, int(t * 1.2), file=fout)

gdef = tf.graph_util.extract_sub_graph(gdef, [opt.node_def.name])
with open("mlp.model", "wb") as fout:
    fout.write(gdef.SerializeToString())
