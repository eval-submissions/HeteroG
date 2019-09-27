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

        shape = tfpb.tensor_shape_pb2.TensorShapeProto()
        for s in size:
            x = shape.dim.add()
            x.size = s

        node.op = 'Const'
        node.attr['dtype'].CopyFrom(tfpb.attr_value_pb2.AttrValue(type=dtype))
        node.attr['value'].CopyFrom(tfpb.attr_value_pb2.AttrValue(tensor={
            'dtype': dtype, 'tensor_shape': shape, 'tensor_content': b'\0' * reduce(operator.mul, size, ts.size)
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
            t = node_def.attr[input_arg.type_attr].type

        for i in range(node_def.attr.get(input_arg.number_attr, 1)):
            type_list.append(t)

    input_nodes = []
    for (i, (t, size)) in enumerate(zip(type_list, node_def.input)):
        size = [int(x) for x in size.split(',') if x] # split gives an empty element if size is empty
        if t >= 100: # need Variable, skip them for now
            print(node_def.op, t)
            return None

        node = graph_def.node.add()
        node.name = 'input_{}'.format(i)
        node.device = node_def.device
        _mkinput(node, t, size)

    profilee = graph_def.node.add()
    profilee.CopyFrom(node_def)

    for i in range(len(node_def.input)):
        profilee.input[i] = 'input_{}'.format(i)

    return graph_def

def _profile(node_def_raw, target):
    if node_def_raw in cache:
        return cache[node_def_raw]

    node_def = tfpb.node_def_pb2.NodeDef()
    node_def.ParseFromString(node_def_raw)

    tf.reset_default_graph()
    gdef = tf.get_default_graph().as_graph_def()

    x = _prepare_graph(node_def, gdef)
    if x == None:
        return 0
    tf.import_graph_def(gdef)

    # TODO: creating ad-hoc sessions introduces a big overhead.
    sess = tf.Session(target)
    run_meta = tf.compat.v1.RunMetadata()
    run_opt = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
    sess.run(tf.get_default_graph().get_operation_by_name('import/profilee'),
        options=run_opt, run_metadata=run_meta)

    for dev in run_meta.step_stats.dev_stats:
        for node in dev.node_stats:
            if node.node_name == 'import/profilee':
                time = node.op_end_rel_nanos - node.op_start_rel_nanos
                # TODO: will there be duplications? A single operation runs on multiple devices (require both host and accelerator)?
                # print(node.op_end_rel_nanos - node.op_start_rel_nanos)

    print("{}: {}".format(node_def.op, time))
    return time

def profiler_factory(target):
    def inner(pointer, size):
        node_def_raw = pointer[:size]
        try:
            return _profile(node_def_raw, target)
        except:
            print("failed")
            return 0
    return inner

# TODO: concurrent profiling if the devices do not overlap? I don't know if tf sessions are thread-safe
# TODO: transfering profiling. currently we just accept a bandwidth matrix (can be generagted using funciton in util) along with the topology
