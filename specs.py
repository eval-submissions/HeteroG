from enum import Enum

class ReplicationMethod(Enum):
    copy, cache, sum, split = range(4)

class ReplicationSpec():
    # split_spec defines given the input replicability, what's the replicability of the outputs (cache or split)
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

        self.split_spec = [([tuplify(x) for x in input], [tuplify(x) for x in output]) for input, output in split_spec]
        self.merge_spec = [([tuplify(x) for x in input], output) for input, output in merge_spec]

def replication_specs():
    copy, cache, sum, split = ReplicationMethod
    cs = (cache, split)
    cc = (copy, cache)
    pure = ([cc], [copy])
    default = ([cs], [cache])
    same = [ ([cache, cs], [cache, ()]), ([cs], [split, ()]) ]
    same2 = [ ([cache, cache, cs], [cache, ()]), ([cs], [split, ()]) ]
    batch = [ ([split, cc], [split]) ]
    batch2 = [ ([cache, split, cc], [split]), ([split, cc], [split]) ]
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
        "Identity": ReplicationSpec([ *same ], [ pure, *batch, ([sum], [sum]) ]),
        "Add": ReplicationSpec([ *same2 ], [ pure, *batch2 ]),
        "Sub": ReplicationSpec([ *same2 ], [ pure, *batch2 ]),
        "Mul": ReplicationSpec([ *same2 ], [ pure, *batch2 ]),
        "Exp": ReplicationSpec([ *same ], [ pure, *batch ]),
        "Neg": ReplicationSpec([ *same ], [ pure, *batch ]),
        "Log1p": ReplicationSpec([ *same ], [ pure, *batch ]),
        "ZerosLike": ReplicationSpec([ *same ], [ pure, *batch ]),
        "GreaterEqual": ReplicationSpec([ *same2 ], [ pure, *batch2 ]),

        # element-wise * N
        "AddN": ReplicationSpec([ ([cache], [cache]), ([split], [split, ()]) ], [ pure, ([split], [split]) ]),

        # nn operations
        "MatMul": ReplicationSpec([ *same ], [ pure, *batch ]),
        "Reciprocal": ReplicationSpec([ *same ], [ pure, *batch ]), # this is not true?
        "BiasAdd": ReplicationSpec([ *same ], [ pure, *batch ]),
        "BiasAddGrad": ReplicationSpec([ default ], [ pure, grad ]),
        "Softmax": ReplicationSpec([ *same ], [ pure, *batch ]),
    }

replication_specs = replication_specs()
