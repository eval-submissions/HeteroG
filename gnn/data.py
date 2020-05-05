import dgl
import re
import numpy as np

class Record:
    def __init__(self, gdef, prof_data, topo):
        self.gdef = gdef
        self.prof_data = prof_data

    def _gen_computation_features(self):
        g = dgl.DGLGraph()
        g.add_nodes(len(self.gdef.node))
        reverse_dict = { node.name: i for i, node in enumerate(self.gdef.node) }
        features = [[np.mean(self.prof_data[(node.name, nrep)]) for nrep in (1, 2, 3, 4)] for node in self.gdef.node]
        for i, node in enumerate(gdef.node):
            for input in node.input:
                if input[0] == '^':
                    x = input[1:]
                else:
                    x = input.split(':')[0]
                g.add_edge(i, reverse_dict[x])
                g.add_edge(reverse_dict[x], i)
        g.add_edges(g.nodes(), g.nodes()) # self-loops are required for GAT
        prof_data = { key: [int((times[0] + times[1]) / 2 * time_ratio) for _, time_ratio in topo["devices"]] for key, times in prof_data.items() }
        return { "gdef": gdef, "prof_data": prof_data, "graph": g, "features": features }


def gen_topo(devices, inter=2810, intra=2810):
    g = dgl.DGLGraph()
    g.add_nodes(len(devices))
    tasks = {}
    for i, (name, time_ratio) in enumerate(devices):
        task = re.search("task:(\d+)/", name)[1]
        if task in tasks:
            for other in tasks[task]:
                g.add_edge(other, i)
                g.add_edge(i, other)
            tasks[task].append(i)
        else:
            tasks[task] = [i]
    g.add_edges(g.nodes(), g.nodes()) # self-loops are required for GAT
    features = [[time_ratio] for _, time_ratio in devices] # currently only this feature
    return { "devices": devices, "graph": g, "features": features, "inter": inter, "intra": intra }

def get_data(gdef, prof_data, topo):

