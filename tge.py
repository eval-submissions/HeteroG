import tensorflow as tf
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
        for name, node in self.name_map.items():
            if hasattr(node, 'replicated'):
                continue

            node.recursive_replicate()

# raw_node: the original node
# inputs/control: linked input and control nodes
# replicated: if this operation is replicated
# spec: the output part of the merge spec. only exists if replicated
# replicability: the output part of the split spec.
# outputs: a list of tensors
class AbstractNode():
    def __init__(self, node, graph):
        self.raw_node = node
        self.graph = graph

        if node.op == 'Const': # always replicate Const. They won't be actaully used if we merge them immediatly, especially if we prune the graph
            self.replicated = True
            self.spec = [ReplicationMethod.copy]

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

    def replicate(self, spec):
        self.replicated = True
        self.spec = spec
    
    # aggresivly replicate a node and all its ancestors
    def recursive_replicate(self):
        try:
            possible_specs = replication_specs[self.raw_node.op].merge_spec
        except KeyError:
            self.replicated = False
            return

        i = 0
        while len(possible_specs) > 0 and i < len(self.inputs):
            input, index = self.inputs[i]
            if not hasattr(input, 'replicated'):
                input.recursive_replicate()
            
            possible_specs = [spec for spec in possible_specs if \
                (input.replicated and get_or_last(input.spec, index) in get_or_last(spec[0], i)) or \
                any(x in get_or_last(spec[0], i) for x in get_or_last(input.replicability, index))]
        
            i += 1

        if len(possible_specs) > 0:
            # prefer splitting
            x = [spec for _,spec in possible_specs if ReplicationMethod.split in spec]
            self.spec = x[0] if len(x) > 0 else possible_specs[0][1]
            self.replicated = True
        else:
            self.replicated = False

class AbstractTensor():
    def __init__(self, node, index):
        self.node = node
        self.index = index

    "get the splited tensor"
    def get_splited(self, id):
        pass

    "get the cached tensor"
    def get_cached(self, id):
        pass

    "return the aggregated tensor"
    def get_merged(self):
        pass


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
