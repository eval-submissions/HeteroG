import tensorflow as tf
from alexnet import alexnet_v2
# outputs, endpoints = alexnet_v2(dataset)


# sess = tf.Session()

def get_device_list():
    from tensorflow.python.client import device_lib
    return [x.name for x in device_lib.list_local_devices()]

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

with tf.device('/gpu:0'):
    dataset = tf.random_uniform([64, 224, 224, 3])

with tf.device('/gpu:1'):
    result = dataset + dataset

graph = tf.get_default_graph()

for node in graph.as_graph_def().node:
    node.device = '/device:CPU:0'

sess.run(result)

tf.Operation

writer = tf.summary.FileWriter('.')
writer.add_graph(tf.get_default_graph())
writer.flush()

tf.import_graph_def()



### 1. get DAG

tf.get_default_graph().as_graph_def().node
# each node is a protobuf structure that has op, name, device, input, attr, etc
# note that `as_graph_def()` is a serilization method: change the returned graph_def do affect the original graph

### 2. add a node

import tensorflow.core.framework.attr_value_pb2

new_node = graph_def.node.add()
new_node.op = "Cast"
new_node.name = "To_Float"
new_node.input.extend(["To_Float"])
new_node.attr["DstT"].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT))
new_node.attr["SrcT"].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT))
new_node.attr["Truncate"].CopyFrom(attr_value_pb2.AttrValue(b=True))

# CopyFrom can also be used to clone a node

### 3. build graph from scrach

# first generate an empty graph_def with
graph_def = tf.Graph().as_graph_def()

# then
graph = tf.Graph()
with graph.as_default():
    tf.graph_util.import_graph_def(graph_def)
