import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32, shape=(None, 1024))
y = tf.placeholder(tf.float32, shape=(None, 10,))
hidden = tf.contrib.slim.fully_connected(x, 256, activation_fn=tf.nn.softmax)
output = tf.contrib.slim.fully_connected(hidden, 10, activation_fn=tf.nn.softmax)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))

dag = optimizer.graph.as_graph_def().node

class AbstractComputationGraph():
    def __init__(self):
        self.name_map = {}
    
    def add_node(self, node):
        self.name_map[node.name] = TaggedNode(node)


class TaggedNode():
    def __init__(self, node):
        self.raw_node = node
        self.input = []
        self.output = []
    

def dstributify(op):
    dag = op.graph.as_graph_def().node

def duplicate_all(n=2):
    new_gd = tf.Graph().as_graph_def()



with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(optimizer, { x: np.random.uniform(size=(10, 1024)), y: np.random.uniform(size=(10, 10)) })

# design:
# 1. for the model part, we need only the optimizer_op


['Tile', 'MatMul', 'RandomUniform', 'Shape', 'Identity', 'Sub',
'Reciprocal', 'Exp', 'ZerosLike', 'AddN', 'Fill', 'Sum', 'VariableV2',
'Reshape', 'Neg', 'BiasAdd', 'Add', 'BiasAddGrad', 'Const', 'Softmax',
'BroadcastGradientArgs', 'Placeholder', 'Select', 'ApplyGradientDescent',
'GreaterEqual', 'Assign', 'NoOp', 'Log1p', 'Mul']