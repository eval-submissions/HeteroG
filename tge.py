import tensorflow as tf
from tensorflow.core.framework.attr_value_pb2 import AttrValue
from tensorflow.core.framework.types_pb2 import *
from specs import *

class AbstractGraph():
    def __init__(self, nodes):
        self.name_map = { node.name: AbstractNode(node, self) for node in nodes }
        for node in self.name_map.values():
            node.link_inputs()
        for node in self.name_map.values():
            node.tag_replicability()

    # replicate all replicable nodes
    def replicate_all(self):
        for node in self.name_map.values():
            if hasattr(node, 'replicated'):
                continue

            node.recursive_replicate()

    def compile_all(self, devices=['/device:CPU:0']):
        self.devices = devices
        self.target = tf.Graph().as_graph_def()
        for node in self.name_map.values():
            node.recursive_compile()

# raw_node: the original node
# inputs/control: linked input and control nodes
# replicated: if this operation is replicated
# spec: the merge spec. only exists if replicated
# replicability: the output part of the split spec.
# outputs: a list of tensors
# replicas: list of replica names. contains only one element if not replicated
class AbstractNode():
    def __init__(self, node, graph):
        self.raw_node = node
        self.graph = graph

        if node.op == 'Const': # always replicate Const. They won't be actaully used if we merge them immediatly, especially if we prune the graph
            self.replicated = True
            self.spec = ([], [ReplicationMethod.copy])

    def tag_replicability(self):
        if self.raw_node.op not in replication_specs:
            self.replicability = [()]
            return

        # find the first matching spec
        for spec in replication_specs[self.raw_node.op].split_spec[:]:
            for i, (input, index) in enumerate(self.inputs):
                if not hasattr(input, 'replicability'):
                    input.tag_replicability()
                if not any(x in get_or_last(spec[0], i) for x in get_or_last(input.replicability, index)):
                    break
            else: # this will also be executed if there is no inputs at all - which is desired
                self.replicability = spec[1]
                return

        self.replicability = [()] # if none of the specs matches, default to not replicable at all

    def link_inputs(self):
        self.inputs = []
        self.control = []
        for input in self.raw_node.input:
            if input[0] == '^':
                self.control.append(input[1:])
            elif ':' in input:
                name, index = input.split(':')
                self.inputs.append((self.graph.name_map[name], int(index)))
            else:
                self.inputs.append((self.graph.name_map[input], 0))

    # aggresivly replicate a node and all its ancestors
    def recursive_replicate(self):
        if hasattr(self, 'replicated'):
            return

        try:
            possible_specs = replication_specs[self.raw_node.op].merge_spec
        except KeyError:
            self.replicated = False
            return

        i = 0
        while len(possible_specs) > 0 and i < len(self.inputs):
            input, index = self.inputs[i]
            input.recursive_replicate()

            possible_specs = [spec for spec in possible_specs if \
                (input.replicated and get_or_last(input.spec[1], index) in get_or_last(spec[0], i)) or \
                any(x in get_or_last(spec[0], i) for x in get_or_last(input.replicability, index))]

            i += 1

        if len(possible_specs) > 0:
            # prefer splitting
            x = [spec for spec in possible_specs if ReplicationMethod.split in spec[1]]
            self.spec = x[0] if len(x) > 0 else possible_specs[0]
            self.replicated = True
        else:
            self.replicated = False

    def recursive_compile(self):
        if hasattr(self, 'replica'):
            return

        for input, index in self.inputs:
            input.recursive_compile()

        if self.replicated:
            n = len(self.graph.devices)
            replicas = [self.graph.target.add() for i in range(n)]
            for id, r in enumerate(replicas):
                r.CopyFrom(self.raw_node)
                r.name += "/replica_{}".format(id)
                for index in range(len(self.inputs)):


        else:
            self.replicas = [self.raw_node]
            pass # setup inputs

    def get_output(self, index):
        if not hasattr(self, 'outputs'):
            self.outputs = []

        while len(self.outputs) <= index:
            self.outputs.append(AbstractTensor(self, len(self.outputs)))

        return self.outputs[index]


class AbstractTensor():
    def __init__(self, node, index):
        self.node = node
        self.index = index

    def get_replicated_split(self, id):
        if not hasattr(self, 'splited'):
            if self.node.replicated:
                if get_or_last(self.node.spec[1], self.index) is ReplicationMethod.split:
                    self.splited = lambda id: "{}:{}".format(self.node.replicas[id], self.index)
                else:
                    raise Exception("not implemented")
            else:
                assert ReplicationMethod.split in get_or_last(self.node.replicability, self.index)
                target = self.node.graph.target
                create_raw_node(target, 'Const', '',
                    "{}/{}".format(self.node.raw_node.name, "aux_split_{}/split_dim".format(self.index)),
                    dtype = { 'type': DT_INT32 },
                    value = { 'tensor': { 'dtype': DT_INT32, 'tensor_shape': {}, 'int_val': [0] } }
                )
                create_raw_node(target, 'Split', '',
                    "{}/{}".format(self.node.raw_node.name, "aux_split_{}".format(self.index)),
                    "{}/{}".format(self.node.raw_node.name, "aux_split_{}/split_dim".format(self.index)),
                    "{}:{}".format(self.node.replicas[0].name, self.index),
                    T = { 'type': DT_FLOAT },
                    num_split = { 'i': len(self.node.graph.devices) }
                )
                self.splited = lambda id: "{}/{}:{}".format(self.node.raw_node.name, "aux_split_{}".format(self.index), id)

            ndevices = len(self.node.graph.devices)

        return self.splited(id)

    def get_replicated_cache(self, id):
        if not hasattr(self, 'cached'):
            if self.node.replicated:
                if get_or_last(self.node.spec[1], self.index) in (ReplicationMethod.cache, ReplicationMethod.copy):
                    self.cached = lambda id: "{}:{}".format(self.node.replicas[id], self.index)
                else:
                    raise Exception("not implemented")
            else:
                assert ReplicationMethod.cache in get_or_last(self.node.replicability, self.index)
                target = self.node.graph.target
                n = len(self.node.graph.devices)
                if 'T' in self.node.replicas[0].attr:
                    dtype = self.node.replicas[0].attr['T']
                elif 'dtype' in self.node.replicas[0].attr:
                    dtype = self.node.replicas[0].attr['dtype']
                else:
                    raise Exception("cannot determine dtype")
                for i in range(n):
                    node = create_raw_node(target, 'Identity', ''
                        "{}/{}".format(self.node.raw_node.name, "aux_cache_{}_{}".format(self.index, i)),
                        "{}:{}".format(self.node.replicas[0].name, self.index),
                    )
                    node.attr['T'].CopyFrom(dtype)
                self.cached = lambda id: "{}/{}".format(self.node.raw_node.name, "aux_cache_{}_{}".format(self.index, id))

        return self.cached(id)

    def get_replicated_copy(self, id):
        assert self.node.replicated and get_or_last(self.node.spec[1], self.index) is ReplicationMethod.copy
        return "{}:{}".format(self.node.replicas[id], self.index)

    def get_replicated_sum(self, id):
        assert self.node.replicated and get_or_last(self.node.spec[1], self.index) is ReplicationMethod.sum
        return "{}:{}".format(self.node.replicas[id], self.index)

    "return the aggregated tensor"
    def get_aggregated(self, id):
        if not self.node.replicated:
            return "{}:{}".format(self.node.replicas[0], self.index)
        switch = {
            ReplicationMethod.copy: self.get_aggregated_copy,
            ReplicationMethod.cache: self.get_aggregated_cache,
            ReplicationMethod.split: self.get_aggregated_split,
            ReplicationMethod.sum: self.get_aggregated_sum
        }
        return switch[get_or_last(self.node.spec[1], self.index)](id)

    def get_aggregated_copy(self, id):
        return "{}:{}".format(self.node.replicas[id], self.index)

    def get_aggregated_cache(self, id):
        raise Exception("this is guarded in `get_aggregated` so never be called")
        return self.get_replicated_cache(self, id)

    def get_aggregated_split(self, id): # TODO: what if two ops on different devices want the same tensor?
        if not hasattr(self, 'merged'):
            target = self.node.graph.target
            device = self.node.devices[id]
            create_raw_node(target, 'Const', device,
                "{}/{}".format(self.node.raw_node.name, "aux_concat_{}/axis".format(self.index)),
                dtype = { 'type': DT_INT32 },
                value = { 'tensor': { 'dtype': DT_INT32, 'tensor_shape': {}, 'int_val': [0] } }
            )
            create_raw_node(target, 'ConcatV2', device,
                "{}/{}".format(self.node.raw_node.name, "aux_concat_{}".format(self.index)),
                *[ "{}:{}".format(replica.name, self.index) for replica in self.node.replicas ],
                "{}/{}".format(self.node.raw_node.name, "aux_concat_{}/axis".format(self.index)),
                N = { 'i': len(self.node.replicas) },
                T = { 'type': DT_FLOAT },
                Tidx = { 'type': DT_INT32 }
            )
            self.merged = "{}/{}".format(self.node.raw_node.name, "aux_concat_{}".format(self.index))

        return self.merged

    def get_aggregated_sum(self, id):
        if not hasattr(self, 'merged'):
            target = self.node.graph.target
            device = self.node.devices[id]
            create_raw_node(target, 'AddN', device,
                "{}/{}".format(self.node.raw_node.name, "aux_sum_{}".format(self.index)),
                *[ "{}:{}".format(replica.name, self.index) for replica in self.node.replicas ],
                N = { 'i': len(self.node.replicas) },
                T = { 'type': DT_FLOAT }
            )
            self.merged = "{}/{}".format(self.node.raw_node.name, "aux_sum_{}".format(self.index))

        return self.merged


def create_raw_node(graph_def, op, device, name, *inputs, **attr):
    node = graph_def.node.add()
    node.name = name
    node.device = device
    node.op = op
    node.input.extend(inputs)
    for name, value in attr.items():
        node.attr[name] = AttrValue(**value)
    return node

# main API: replace an op with another op which is backed by a distributed graph
def distributify(op):
    dag = op.graph.as_graph_def()
    # prune the graph
    # init = tf.global_variables_initializer()
    # dag = tf.graph_util.extract_sub_graph(dag, [op.name, init.name])
    acg = AbstractGraph(dag.node)
    acg.tag()
    dag = acg.assemble()
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(dag)
    return graph

def get_or_last(list, x):
    return list[x] if x < len(list) else list[-1]

# design
# naming scheme: for replicas, <origin name>/replica_x where x is the device id. for auxiliary nodes, <parent node>/aux_x where x is a local increament number


# constraints
# 1. currently we require all inputs are replicated, this prevents operation that has one input already replicated while other need to be all-reduced
# 2. most specs (except for identity) currently only preserve batchiness, if one apply operators to gradients (have the merge type "sum"), they won't get replicated
