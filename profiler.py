import os
import time
import tensorflow as tf
import tensorflow.core.framework as tfpb
import operator
from functools import reduce
from http.server import HTTPServer, BaseHTTPRequestHandler

cache = {} # TODO: LRU?

import utils
op_def_dict = utils.op_def_dict()

def _mkinput(node, dtype, size):
    if dtype < 100:
        ts = tf.DType(dtype)
        if not ts.is_integer and not ts.is_floating: # complex, string, or other unhandleable fancy things
            raise Exception("not implemented")

        node.op = 'Const'
        node.attr['dtype'].CopyFrom(tfpb.attr_value_pb2.AttrValue(type=dtype))
        node.attr['value'].CopyFrom(tfpb.attr_value_pb2.AttrValue(tensor={
            'dtype': dtype, 'tensor_shape': { 'dim': size }, 'tensor_content': [0] * reduce(operator.mul, size, ts.size)
        }))
    else:
        node.op = 'VariableV2'
        raise Exception("not implemented")

# node_def is a normal node def but instead of the name, the inputs are replaced by the sizes seperated by commas (and empty name means scalar)
def _prepare_graph(node_def, graph_def):
    type_list = []
    op_def = op_def_dict[node_def.op]
    for input_arg in op_def.input_arg:
        if input_arg.type:
            t = input_arg.type
        elif input_arg.type_attr:
            t = node_def.attr[input_arg.type_attr]

        for i in range(node_def.attr.get(input_arg.number_attr, 1)):
            type_list.append(t)

    input_nodes = []
    for (i, (t, size)) in enumerate(zip(type_list, node_def.input)):
        size = [int(x) for x in size.split(',') if x] # split gives an empty element if size is empty
        if t >= 100: # need Variable, skip them for now
            print(node_def.op, t)
            return 0

        node = graph_def.node.add()
        node.name = 'input_{}'.format(i)
        node.device = node_def.device
        _mkinput(node, t, size)

    profilee = graph_def.node.add()
    profilee.CopyFrom(node_def)
    profilee.input = ['input_{}'.format(i) for i in range(node_def.input)]

    return gdef

def profile(node_def_raw, target):
    if node_def_raw in cache:
        return cache[node_def_raw]

    node_def = tfpb.node_def_pb2.NodeDef()
    node_def.ParseFromString(node_def_raw)

    tf.reset_default_graph()
    gdef = tf.get_default_graph().as_graph_def()

    _prepare_graph(gdef)
    tf.import_graph_def(gdef)

    sess = tf.Session(target)
    run_meta = tf.compat.v1.RunMetadata()
    run_opt = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
    sess.run(tf.get_default_graph().get_operation_by_name('import/profilee'),
        options=run_opt, run_metadata=run_meta)




# whether this script should be run in an independent process is not decided
# independent: they are natually decoupled, the result can be used for to several runs
# not independent: they share the same tf session which is nice


# class Handler(BaseHTTPRequestHandler):
#     def do_POST(self):
#         buf = self.rfile.read() # the name field should always be called "profilee" so we can match the cache by bytes

#         node_def = tfpb.node_def_pb2.NodeDef()
#         node_def.ParseFromString(buf)

#         profile

# try:
#     # parse commandline to get device information
#     HTTPServer(('0.0.0.0', 3907), Handler).serve_forever()
# except KeyboardInterrupt:
#     print("bye~")

# TODO: concurrent profiling if the devices do not overlap? I don't know if tf sessions are thread-safe
# TODO: transfering profiling. currently we just accept a bandwidth matrix (can be generagted using funciton in util) along with the topology
