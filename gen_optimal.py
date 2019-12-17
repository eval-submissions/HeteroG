import numpy as np
import tensorflow as tf

def model_fn():
    x = tf.placeholder(tf.float32, shape=(256, 102400))
    y = tf.placeholder(tf.float32, shape=(256, 100,))
    w1 = tf.Variable(tf.random_normal([102400, 2560]))
    b = tf.Variable(tf.zeros([2560]))
    h = tf.matmul(x, w1) + b # 256, 2560
    w2 = tf.Variable(tf.random_normal([2560, 100]))
    o = tf.matmul(h, w2) # 256, 100
    l = tf.reduce_sum(o - y) # 256, 100
    go = l * o # 256, 100
    gw2 = tf.matmul(tf.transpose(h), go) # 2560, 100
    gh = tf.matmul(go, tf.transpose(w2)) # 256, 2560
    gw1 = tf.matmul(tf.transpose(x), gh) # 102400, 2560
    gb = tf.reduce_sum(gh, axis=[0]) # 2560
    opt = tf.train.GradientDescentOptimizer(0.2)
    opt = opt.apply_gradients([(gw2, w2), (gw1, w1), (gb, b)])
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
        print(node.name, p.profile(node.name, "/gpu:0"), file=fout)

gdef = tf.graph_util.extract_sub_graph(gdef, [opt.node_def.name])
with open("mlp.model", "wb") as fout:
    fout.write(gdef.SerializeToString())
