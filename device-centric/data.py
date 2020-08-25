import dgl
import re
import numpy as np
import pickle
import math
import itertools
import networkx as nx
from utils import groupby, car, cadr, cdr

import sys
def info(*args):
    print(*args, file=sys.stdout, flush=True)

def gen_topo(devices, inter=2810, intra=2810):
    g = dgl.DGLGraph()
    g.add_nodes(len(devices))
    nfeats = [[time_ratio / 10, memory / 10_000_000_000] for _, time_ratio, memory in devices]
    efeats = []
    tasks = {}
    for i, (name, *_) in enumerate(devices):
        task = re.search("task:(\d+)/", name)[1]
        if task in tasks:
            for other in tasks[task]:
                g.add_edge(other, i)
                g.add_edge(i, other)
                efeats.append([0, intra / 100_000, math.log(intra) / 10])
                efeats.append([0, intra / 100_000, math.log(intra) / 10])
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
                        efeats.append([1, inter / 100_000, math.log(inter) / 10])
                        efeats.append([1, inter / 100_000, math.log(inter) / 10])
    bandwidth = [x for _, x, _ in efeats]
    groups = k_spanning_tree(g, bandwidth, 2) + k_spanning_tree(g, bandwidth, 4) + [[0]] + [list(range(len(devices)))]

    base_nccl_model = [0.043420241077615454, 368.2013618677043, 0.27766802543921265, 211.91926070037152]
    nccl_models = {}
    dgroups = groupby(devices, key=lambda x: re.search("task:(\d+)/", x[0])[1], value=lambda x: x[0])

    for task, devs in dgroups.items():
        nccl_models[','.join(sorted(devs))] = [ x * 2810 / intra for x in base_nccl_model ]

    for tasks in (t for i in range(2, len(dgroups)+1) for t in itertools.combinations(dgroups.keys(), i)):
        devs = [dgroups[t][0] for t in tasks] # the first (alphabet order) device is the leader of the task
        nccl_models[','.join(sorted(devs))] = [ x * 2810 / inter for x in base_nccl_model ]

    return { "devices": devices, "graph": g, "nfeats": nfeats, "efeats": efeats, "groups": groups, "inter": inter, "intra": intra, "nccl_models": nccl_models }

def gen_data(gdef, prof_data, topo, op_table):
    g = dgl.DGLGraph()
    g.add_nodes(len(gdef.node))
    ntypes = []
    for node in gdef.node:
        if node.op not in op_table:
            op_table[node.op] = len(op_table)
        ntypes.append(op_table[node.op])
    nfeats = [[np.mean(prof_data[(node.name, nrep)]) / 10_000 for nrep in (1, 2, 4, 8)] for node in gdef.node]
    efeats = []
    reverse_dict = { node.name: i for i, node in enumerate(gdef.node) }
    for i, node in enumerate(gdef.node):
        for input in node.input:
            if input[0] == '^':
                x = input[1:]
                input_index = 0
            else:
                x = input.split(':')[0]
                try:
                    input_index = int(input.split(':')[1])
                except:
                    input_index = 0
            g.add_edge(i, reverse_dict[x])
            g.add_edge(reverse_dict[x], i)
            try:
                shape = [ dim.size for dim in gdef.node[1].attr["_output_shapes"].list.shape[input_index] ]
                if shape[0] == -1:
                    shape[0] = 120
                tensorsize = 1
                for size in shape:
                    if size == -1:
                        p = 0
                        break
                    tensorsize *= size
            except:
                tensorsize = -1
            efeats.append([1, tensorsize / 100_000_000])
            efeats.append([-1, tensorsize / 100_000_000])
    prof_data = { key: [int(np.mean(times) * time_ratio) for _, time_ratio, _ in topo["devices"]] for key, times in prof_data.items() }

    def group_with_layer_name():
        group_table = {}
        for i, node in enumerate(gdef.node):
            if node.name.startswith("GradientDescent") or node.name.startswith("gradients"):
                prefix = '/'.join(node.name.split('/')[1:3])
            else:
                prefix = '/'.join(node.name.split('/')[:2])
            if prefix in group_table:
                group_table[prefix].append(i)
            else:
                group_table[prefix] = [i]
        return list(group_table.values())

    def group_with_k_spanning_tree():
        seed = [i for i, node in enumerate(gdef.node) if node.name == 'GradientDescent'][0]
        return k_spanning_tree(g, [x[1] for x in efeats], 20, seed)

    def group_with_topk_nodes():
        from utils import group_around_topk_costs
        from tge import TGE

        base_groups = TGE(gdef, [dev for dev, _, _ in topo["devices"]]).get_groups()
        id_list = group_around_topk_costs(gdef, base_groups, prof_data, 19)
        return list(groupby(enumerate(id_list), key=cadr, value=car).values())

    cgroups = group_with_topk_nodes()

    # for group in cgroups:
    #     for g in group:
    #         info(gdef.node[g].name)
    #     info()

    # out_of_group = []
    # for i in range(len(gdef.node)):
    #     for g in cgroups:
    #         if i in g:
    #             break
    #     else:
    #         out_of_group.append(i)
    # info(cgroups, out_of_group)

    return {
        "gdef": gdef,
        "prof_data": prof_data,
        "devices": topo["devices"],
        "cgraph": g,
        "cgroups": cgroups,
        "cnfeats": nfeats,
        "cntypes": ntypes,
        "cefeats": efeats,
        "tgraph": topo["graph"],
        "tgroups": None, # topo["groups"],
        "tnfeats": topo["nfeats"],
        "tefeats": topo["efeats"],
        "op_table": op_table,
        # the two are workarounds; should write a graph parser in tge.py to get the links and bandwidth from graph
        "inter": topo["inter"],
        "intra": topo["intra"],
        "nccl_models": topo["nccl_models"]
    }

def get_all_data():
    models = [pickle.load(open("{}.pickle".format(m), "rb")) for m in ("vgg", "resnet", "inception", )] #  , "mlp", "lenet"
    topos1 = [gen_topo([
        ("/job:worker/replica:0/task:0/device:GPU:0", 1, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:1", 1, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:2", 1, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:3", 1, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:4", 1.5, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:5", 1.5, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:6", 1.5, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:7", 1.5, 3<<30),
    ], intra=bandwidth) for bandwidth in (1000, 10000)]
    topos2 = [gen_topo([
        ("/job:worker/replica:0/task:0/device:GPU:0", 1, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:1", 1.5, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:2", 2, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:3", 2.5, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:4", 3, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:5", 3.5, 3<<30),
    ], intra=bandwidth) for bandwidth in (400, 4000, 40000)]
    topos3 = [gen_topo([
        ("/job:worker/replica:0/task:0/device:GPU:0", 1, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:1", 1, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:2", 1, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:3", 1, 3<<30),
        ("/job:worker/replica:0/task:1/device:GPU:0", 1, 3<<30),
        ("/job:worker/replica:0/task:1/device:GPU:1", 1, 3<<30),
    ], intra=bandwidth, inter=1000) for bandwidth in (4000, 40000)]
    op_table = {}
    return [gen_data(gdef, prof_data, topo, op_table) for gdef, prof_data in models for topo in topos1 + topos2 + topos3]

# prim's algorithm
# alternative: https://networkx.github.io/documentation/stable/reference/algorithms/tree.html#module-networkx.algorithms.tree.mst
def k_spanning_tree(g, weights, k, seed=0):
    def get_weight(center, neighbor):
        return weights[ng.adj[center][neighbor][0]['id']]

    ng = g.to_networkx()
    tree_nodes = [seed]
    tree_edges = []
    while True:
        bridges = [(center, neighbor) for center in tree_nodes for neighbor in ng.adj[center] if neighbor not in tree_nodes ]
        if len(bridges) == 0:
            break
        highest_weight = np.max([ get_weight(center, neighbor) for center, neighbor in bridges ])
        index_of_edge_to_add = np.random.choice([ i for i, (center, neighbor) in enumerate(bridges) if get_weight(center, neighbor) == highest_weight ])
        center, neighbor = bridges[index_of_edge_to_add]
        tree_nodes.append(neighbor)
        tree_edges.append((center, neighbor, highest_weight))
    tree_edges.sort(key=lambda x: x[2])
    tree_edges = set( (center, neighbor) for center, neighbor, weight in tree_edges[k-1:] )
    groups = []
    for node in tree_nodes:
        for group in groups:
            for neighbor in group:
                if (node, neighbor) in tree_edges or (neighbor, node) in tree_edges:
                    group.append(node)
                    break
            else:
                continue
            break
        else:
            groups.append([node])

    return groups
