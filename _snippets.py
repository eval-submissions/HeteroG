####################
# 0. get device list
####################

def get_device_list():
    from tensorflow.python.client import device_lib
    return [x.name for x in device_lib.list_local_devices()]

##############
### 1. get DAG
##############

tf.get_default_graph().as_graph_def().node
# each node is a protobuf structure that has op, name, device, input, attr, etc
# note that `as_graph_def()` is a serilization method: change the returned graph_def do affect the original graph

#################
# 2. add a node
#################

import tensorflow.core.framework.attr_value_pb2

new_node = graph_def.node.add()
new_node.op = "Cast"
new_node.name = "To_Float"
new_node.input.extend(["To_Float"])
new_node.attr["DstT"].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT))
new_node.attr["SrcT"].CopyFrom(attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT))
new_node.attr["Truncate"].CopyFrom(attr_value_pb2.AttrValue(b=True))

# CopyFrom can also be used to clone a node

##############################
# 3. build graph from scrach
##############################

# first generate an empty graph_def with
graph_def = tf.Graph().as_graph_def()

# then
graph = tf.Graph()
with graph.as_default():
    tf.graph_util.import_graph_def(graph_def)

##############################
# 4. show graph in tensorboard
##############################
def show(graph):
    writer = tf.summary.FileWriter('.')
    writer.add_graph(graph)
    writer.flush()
