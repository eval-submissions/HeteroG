import dgl
import re
import numpy as np
import pickle

def gen_topo(devices, inter=2810, intra=2810):
    g = dgl.DGLGraph()
    g.add_nodes(len(devices))
    nfeats = [[time_ratio, memory / 10000000000] for _, time_ratio, memory in devices]
    efeats = []
    tasks = {}
    for i, (name, *_) in enumerate(devices):
        task = re.search("task:(\d+)/", name)[1]
        if task in tasks:
            for other in tasks[task]:
                g.add_edge(other, i)
                g.add_edge(i, other)
                efeats.append([0, intra / 10000])
                efeats.append([0, intra / 10000])
            tasks[task].append(i)
        else:
            tasks[task] = [i]
    for task, devs in tasks.items():
        for dev in devs:
            for another_task, other_devs in tasks.items():
                if another_task != task:
                    for another_dev in other_devs:
                        g.add_edge(dev, another_dev)
                        g.add_edge(another_dev, dev)
                        efeats.append([1, inter / 10000])
                        efeats.append([1, inter / 10000])
    return { "devices": devices, "graph": g, "nfeats": nfeats, "efeats": efeats, "inter": inter, "intra": intra }

def gen_data(gdef, prof_data, topo):
    g = dgl.DGLGraph()
    g.add_nodes(len(gdef.node))
    nfeats = [[np.mean(prof_data[(node.name, nrep)]) for nrep in (1, 2, 3, 4)] for node in gdef.node]
    efeats = []
    reverse_dict = { node.name: i for i, node in enumerate(gdef.node) }
    for i, node in enumerate(gdef.node):
        for input in node.input:
            if input[0] == '^':
                x = input[1:]
            else:
                x = input.split(':')[0]
            g.add_edge(i, reverse_dict[x])
            g.add_edge(reverse_dict[x], i)
            efeats.append([1]) # TODO: tensorsize. Note the batchsize need to be given somewhere
            efeats.append([-1])
    prof_data = { key: [int(np.mean(times) * time_ratio / 1000) for _, time_ratio, _ in topo["devices"]] for key, times in prof_data.items() }
    return {
        "gdef": gdef,
        "prof_data": prof_data,
        "devices": topo["devices"],
        "cgraph": g,
        "cnfeats": nfeats,
        "cefeats": efeats,
        "tgraph": topo["graph"],
        "tnfeats": topo["nfeats"],
        "tefeats": topo["efeats"],
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
    ], intra=bandwidth) for bandwidth in (200, 2000)]
    topos3 = [gen_topo([
        ("/job:worker/replica:0/task:0/device:GPU:0", 1, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:1", 1, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:2", 1.2, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:3", 1.2, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:4", 1.5, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:5", 1.5, 6<<30),
    ], intra=bandwidth) for bandwidth in (400, 4000)]
    topos4 = [gen_topo([
        ("/job:worker/replica:0/task:0/device:GPU:0", 1, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:1", 1, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:2", 1, 6<<30),
        ("/job:worker/replica:0/task:0/device:GPU:3", 1, 6<<30),
        ("/job:worker/replica:0/task:1/device:GPU:0", 1, 6<<30),
        ("/job:worker/replica:0/task:1/device:GPU:1", 1, 6<<30),
    ], intra=bandwidth, inter=100) for bandwidth in (100, 1000, 10000)]
    return [gen_data(gdef, prof_data, topo) for gdef, prof_data in models for topo in topos1 + topos2 + topos3 + topos4]
