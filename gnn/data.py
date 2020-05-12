import dgl
import re
import numpy as np
import pickle

def gen_topo(devices, inter=2810, intra=2810):
    g = dgl.DGLGraph()
    g.add_nodes(len(devices))
    tasks = {}
    for i, (name, *_) in enumerate(devices):
        task = re.search("task:(\d+)/", name)[1]
        if task in tasks:
            for other in tasks[task]:
                g.add_edge(other, i)
                g.add_edge(i, other)
            tasks[task].append(i)
        else:
            tasks[task] = [i]
    g.add_edges(g.nodes(), g.nodes()) # self-loops are required for GAT
    features = [[time_ratio, memory, intra] for _, time_ratio, memory in devices]
    return { "devices": devices, "graph": g, "features": features, "inter": inter, "intra": intra }

def gen_data(gdef, prof_data, topo):
    g = dgl.DGLGraph()
    g.add_nodes(len(gdef.node))
    reverse_dict = { node.name: i for i, node in enumerate(gdef.node) }
    features = [[np.mean(prof_data[(node.name, nrep)]) for nrep in (1, 2, 3, 4)] for node in gdef.node]
    for i, node in enumerate(gdef.node):
        for input in node.input:
            if input[0] == '^':
                x = input[1:]
            else:
                x = input.split(':')[0]
            g.add_edge(i, reverse_dict[x])
            g.add_edge(reverse_dict[x], i)
    g.add_edges(g.nodes(), g.nodes()) # self-loops are required for GAT
    prof_data = { key: [int(np.mean(times) * time_ratio) for _, time_ratio, _ in topo["devices"]] for key, times in prof_data.items() }
    return {
        "gdef": gdef,
        "prof_data": prof_data,
        "computation_graph": g,
        "computation_features": features,
        "devices": topo["devices"],
        "device_graph": topo["graph"],
        "device_features": topo["features"],
        # the two are workarounds; should write a graph parser in tge.py to get the links and bandwidth from graph
        "inter": topo["inter"],
        "intra": topo["intra"]
    }

def get_all_data():
    models = [pickle.load(open("{}.pickle".format(m), "rb")) for m in ("vgg", )] # "resnet", "mlp", "lenet"
    topos1 = [gen_topo([
        ("/job:worker/replica:0/task:0/device:GPU:0", 1, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:1", 1, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:2", 1, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:3", 1, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:4", 1.2, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:5", 1.2, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:6", 1.2, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:7", 1.2, 6<<30),
    ], intra=bandwidth) for bandwidth in (10, 100, 1000, 10000)]
    topos2 = [gen_topo([
        ("/job:worker/replica:0/task:0/device:GPU:0", 1.5, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:1", 1.2, 6<<30),
    ], intra=bandwidth) for bandwidth in (20, 200, 2000)]
    topos3 = [gen_topo([
        ("/job:worker/replica:0/task:0/device:GPU:0", 1, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:1", 1, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:2", 1.2, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:3", 1.2, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:4", 1.5, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:5", 1.5, 6<<30),
    ], intra=bandwidth) for bandwidth in (40, 400, 4000)]
    return [gen_data(gdef, prof_data, topo) for gdef, prof_data in models for topo in topos1 + topos2 + topos3]
