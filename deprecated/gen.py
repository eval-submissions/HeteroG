def model_fn(bsize=None):
    from tensorflow.contrib.slim.nets import vgg
    x = tf.placeholder(tf.float32, shape=(bsize, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(bsize, 1000))
    output, _ = vgg.vgg_19(x, 1000)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
    return optimizer

import tensorflow as tf
import pickle

opt = model_fn()
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def(add_shapes=True)

devices = (
    "/job:tge/replica:0/task:0/device:GPU:0",
    "/job:tge/replica:0/task:1/device:GPU:0"
)

import tge

g = (tge.TGE(gdef, devices)
    .custom({ node.name: [1, 1, 1] for node in gdef.node })
    # .replace_placeholder(64)
    # .use_nccl()
    .verbose()
    .compile()
    .get_result()
)

pickle.dump(g, open("model.pickle", "wb"))
