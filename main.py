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
    
    def tag(self):
        pass


class TaggedNode():
    def __init__(self, node):
        self.raw_node = node


# main API: replace an op with another op which is backed by a distributed graph
def distributify(op):
    dag = op.graph.as_graph_def()
    # prune the graph
    # init = tf.global_variables_initializer()
    # dag = tf.graph_util.extract_sub_graph(dag, [op.name, init.name])
    acg = AbstractComputationGraph()
    for node in dag.node:
        acg.add_node(node)
    acg.tag()
    dag = acg.assemble()
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(dag)
    return graph

def duplicate_all(n=2):
    new_gd = tf.Graph().as_graph_def()



with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(optimizer, { x: np.random.uniform(size=(10, 1024)), y: np.random.uniform(size=(10, 10)) })

class ReplicationMethod(Enum):
    none, copy, cache, sum, fetch, split = range(6)

class ReplicationSpec():
    # split_spec defines if this operator is not replicated, given the input replicability, what's the replicability of the outputs [none, cache, split]
    # merge_spec describes when this operator can be replicated, and how the outputs can be merged [copy, cache, sum, split]
    # if the inputs does not match any merge_spec, this operator won't be replicated
    # if the inputs does not match any split_spec, the outputs won't be replicated (so the children also won't be replicated)
    # both inputs and outputs lists can be seen as infinitive list repeating the last element.
    def __init__(self, split_spec=[], merge_spec=[]):
        def tuplify(x):
            if not isinstance(x, tuple):
                return (x,)
            else:
                return x

        self.split_spec = [([tuplify(x) for x in input], (tuplify(x) for x in output)) for input, output in split_spec]
        self.merge_spec = [([tuplify(x) for x in input], (tuplify(x) for x in output)) for input, output in merge_spec]

def replication_specs():
    none, copy, cache, sum, fetch, split = ReplicationMethod
    cs = (cache, split)
    cc = (copy, cache)
    pure = ([cc], [copy])
    default = ([cs], [cache])
    same = [ ([cache, cs], [cache, none]), ([split, cs], [split, none]) ]
    same2 = [ ([cache, cache, cs], [cache, cache, none]), ([split, split, cs], [split, none]) ] # TODO: require only one split to allow split?
    batch = ([split, cc], [split])
    batch2 = ([split, split, cc], [split]) # TODO: require only one split to allow split?
    grad = ([split, cc], [sum])
    return {
        # special
        "VariableV2": ReplicationSpec([ default ]),
        "Placeholder": ReplicationSpec([ ([], [cs]) ]),
        "Const": ReplicationSpec([ default ]), # by default, this spec means Const never get replicated. Special case for Const is implemented in the tagging funciton
        "Reshape": ReplicationSpec([ default ], [ pure, ([split, sum], [split]) ]), # need reconsideration
        "Shape": ReplicationSpec([ default ], [ pure, ([split], [sum]) ]), # Shape and Tile work together should pass down the split-ability.
        "Tile": ReplicationSpec([ *same ], [ pure, ([cc, sum], [split]), ([split, cc], [split]) ]),
        "Fill": ReplicationSpec([ default ], [ pure, ([sum, cc], [split]) ]),
        "BroadcastGradientArgs": ReplicationSpec([ default ], [ pure, ([sum], [copy]) ]), # this undocumented operator accepts two shapes and returns the dimension that should be reduced
        "Select": ReplicationSpec([ ([cache, cache, cache, cs], [cache]), ([cs], [split]) ], [ pure, ([(copy, cache, split)], [split]) ]),
        "NoOp": ReplicationSpec([], [([(copy, cache, sum, split)], [cache])]), # replicate this might reduce the signal transmission?

        # ignore (cannot replicate)
        # ApplyGradientDescent
        # Assign
        
        # element-wise
        "Identity": ReplicationSpec([ *same ], [ pure, batch ]),
        "Add": ReplicationSpec([ *same2 ], [ pure, batch2 ]),
        "Sub": ReplicationSpec([ *same2 ], [ pure, batch2 ]),
        "Mul": ReplicationSpec([ *same2 ], [ pure, batch2 ]),
        "Exp": ReplicationSpec([ *same ], [ pure, batch ]),
        "Neg": ReplicationSpec([ *same ], [ pure, batch ]),
        "Log1p": ReplicationSpec([ *same ], [ pure, batch ]),
        "ZerosLike": ReplicationSpec([ *same ], [ pure, batch ]),
        "GreaterEqual": ReplicationSpec([ *same2 ], [ pure, batch2 ]),

        # element-wise * N
        "AddN": ReplicationSpec([ ([cache], [cache]), ([split], [split, none]) ], [ pure, ([split], [split]) ]),

        # nn operations
        "MatMul": ReplicationSpec([ *same ], [ pure, batch ]),
        "Reciprocal": ReplicationSpec([ *same ], [ pure, batch ]), # this is not true?
        "BiasAdd": ReplicationSpec([ *same ], [ pure, batch ]),
        "BiasAddGrad": ReplicationSpec([ default ], [ pure, grad ]),
        "SoftMax": ReplicationSpec([ *same ], [ pure, batch ]),
    }

replication_specs = replication_specs()

# constraints
# 1. currently we require all inputs are replicated, this prevents operation that has one input already replicated while other need to be all-reduced
