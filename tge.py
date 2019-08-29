from specs import *

class AbstractComputationGraph():
    def __init__(self, nodes):
        self.name_map = { node.name: TaggedNode(node) for node in nodes }
        for node in self.name_map.values():
            node.link_inputs(self.name_map)

    def tag(self):
        pass

    def replicate_all(self):
        for name, node in self.name_map:
            if hasattr(node, 'replicate'):
                continue

            # try to replicated
            if node.raw_node.op == 'Const':
                node.replicate()
            else:
                possible_specs = replication_specs[node.raw_node.op].merge_spec
                for i, (input, index) in enumerate(node.raw_node.input):
                    if not hasattr(input, 'replication'):
                        input.replicate()



            if not hasattr(node, 'replicate'):
                self.replicate(node)

    def replicate(self, node):
        pass
            
# naming scheme: for replicas, <origin name>/replica_x where x is the device id
# for auxiliary nodes, <parent node>/aux_x where x is a local increament number



# raw_node: the original node
# inputs/control: linked input and control nodes
# replicated: if this operation is replicated
# spec: the output part of spec. This field exists if and only if `replicated` exists
# outputs: a list of output info (replicated, split/merge method)
class TaggedNode():
    def __init__(self, node):
        self.raw_node = node

        if node.op == 'Const': # always replicate Const. They won't be actaully used if we merge them immediatly, especially if we prune the graph
            self.replicated = True

        self._replicability_split = None
        self._replicability_merge = None

    def link_inputs(self, name_map):
        self.inputs = []
        self.control = []
        for input in self.raw_node.input:
            if input[0] == '^':
                self.control.append(input[1:])
            elif ':' in input:
                name, index = input.split(':')
                self.inputs.append((name_map[name], int(index)))
            else:
                self.inputs.append((name_map[name], 0))

    def replicability_split(self, inputs):
        if self._replicability is not None:
            return self._replicability
        
# main API: replace an op with another op which is backed by a distributed graph
def distributify(op):
    dag = op.graph.as_graph_def()
    # prune the graph
    # init = tf.global_variables_initializer()
    # dag = tf.graph_util.extract_sub_graph(dag, [op.name, init.name])
    acg = AbstractComputationGraph(dag.node)
    acg.tag()
    dag = acg.assemble()
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(dag)
    return graph

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(optimizer, { x: np.random.uniform(size=(10, 1024)), y: np.random.uniform(size=(10, 10)) })


def get_or_last(list, x):
    if x < len(list):
        return list[x]
    else:
        return list[-1]

# constraints
# 1. currently we require all inputs are replicated, this prevents operation that has one input already replicated while other need to be all-reduced
# 2. most specs (except for identity) currently only pass down batchiness, if one apply operators to gradients (have the merge type "sum"), they won't get replicated
