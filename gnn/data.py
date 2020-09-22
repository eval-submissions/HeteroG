import dgl
import re
import numpy as np
import pickle
import math
import itertools
import networkx as nx
from utils import groupby, car, cadr, cdr, info

def gen_data(gdef, prof_data, op_table, devices, inter=2810, intra=2810):
    edge_link = [], []
    link_feats = []
    device_feats = [[time_ratio / 10, memory / 10_000_000_000] for _, time_ratio, memory in devices]
    tasks = {}
    for i, (name, *_) in enumerate(devices):
        task = re.search("task:(\d+)/", name)[1]
        if task in tasks:
            for other in tasks[task]:
                edge_link[0].append(i)
                edge_link[1].append(other)
                edge_link[0].append(other)
                edge_link[1].append(i)
                link_feats.append([0, intra / 100_000, math.log(intra) / 10])
                link_feats.append([0, intra / 100_000, math.log(intra) / 10])
            tasks[task].append(i)
        else:
            tasks[task] = [i]
    for task, devs in tasks.items():
        for dev in devs:
            for another_task, other_devs in tasks.items():
                if another_task != task:
                    for another_dev in other_devs:
                        edge_link[0].append(dev)
                        edge_link[1].append(another_dev)
                        edge_link[0].append(another_dev)
                        edge_link[1].append(dev)
                        link_feats.append([1, inter / 100_000, math.log(inter) / 10])
                        link_feats.append([1, inter / 100_000, math.log(inter) / 10])
    # bandwidth = [x for _, x, _ in link_feats]
    # tgroups = k_spanning_tree(g, bandwidth, 2) + k_spanning_tree(g, bandwidth, 4) + [[0]] + [list(range(len(devices)))]

    base_nccl_model = [0.043420241077615454, 368.2013618677043, 0.27766802543921265, 211.91926070037152]
    nccl_models = {}
    dgroups = groupby(devices, key=lambda x: re.search("task:(\d+)/", x[0])[1], value=lambda x: x[0])

    for task, devs in dgroups.items():
        nccl_models[','.join(sorted(devs))] = [ x * 2810 / intra for x in base_nccl_model ]

    for tasks in (t for i in range(2, len(dgroups)+1) for t in itertools.combinations(dgroups.keys(), i)):
        devs = [dgroups[t][0] for t in tasks] # the first (alphabet order) device is the leader of the task
        nccl_models[','.join(sorted(devs))] = [ x * 2810 / inter for x in base_nccl_model ]

    op_types = []
    for node in gdef.node:
        if node.op not in op_table:
            op_table[node.op] = len(op_table)
        op_types.append(op_table[node.op])
    op_feats = [[np.mean(prof_data[(node.name, nrep)]) / 10_000 for nrep in (1, 2, 4, 8)] for node in gdef.node]
    tensor_feats = []
    edge_prev = ([], [])
    edge_succ = ([], [])
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
            edge_prev[0].append(i)
            edge_prev[1].append(reverse_dict[x])
            edge_succ[0].append(reverse_dict[x])
            edge_succ[1].append(i)
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
            tensor_feats.append([tensorsize / 100_000_000])
    prof_data = { key: [int(np.mean(times) * time_ratio) for _, time_ratio, _ in devices] for key, times in prof_data.items() }

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

        base_groups = TGE(gdef, [dev for dev, _, _ in devices]).get_groups()
        id_list = group_around_topk_costs(gdef, base_groups, prof_data, 19)
        return list(groupby(enumerate(id_list), key=cadr, value=car).values())

    op_groups = group_with_topk_nodes()

    # out_of_group = []
    # for i in range(len(gdef.node)):
    #     for g in cgroups:
    #         if i in g:
    #             break
    #     else:
    #         out_of_group.append(i)
    # info(cgroups, out_of_group)

    edge_place = ([], [])
    edge_serve = ([], [])
    for i in range(len(gdef.node)):
        for j in range(len(devices)):
            edge_place[0].append(i)
            edge_place[1].append(j)
            edge_serve[0].append(j)
            edge_serve[1].append(i)

    g = dgl.heterograph({
        ('device', 'link', 'device'): edge_link,
        ('op', 'prev', 'op'): edge_prev,
        ('op', 'succ', 'op'): edge_succ,
        ('op', 'place', 'device'): edge_place,
        ('device', 'serve', 'op'): edge_serve
    })

    return {
        "graph": g,
        "gdef": gdef,
        "prof_data": prof_data,
        "devices": devices,
        "op_groups": op_groups,
        "device_groups": None, # groups,
        "op_feats": op_feats,
        "op_types": op_types,
        "device_feats": device_feats,
        "tensor_feats": tensor_feats,
        "link_feats": link_feats,
        "op_table": op_table,
        # the two are workarounds; should write a graph parser in tge.py to get the links and bandwidth from graph
        "inter": inter,
        "intra": intra,
        "nccl_models": nccl_models
    }

def get_all_data():
    models = [pickle.load(open("{}.pickle".format(m), "rb")) for m in ("vgg", "resnet", "inception", )] #  , "mlp", "lenet"
    topo_spec1 = [([
        ("/job:worker/replica:0/task:0/device:GPU:0", 1, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:1", 1, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:2", 1, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:3", 1, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:4", 1.5, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:5", 1.5, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:6", 1.5, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:7", 1.5, 3<<30),
    ], 2810, bandwidth) for bandwidth in (1000, 10000)]
    topo_spec2 = [([
        ("/job:worker/replica:0/task:0/device:GPU:0", 1, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:1", 1.5, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:2", 2, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:3", 2.5, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:4", 3, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:5", 3.5, 3<<30),
    ], 2810, bandwidth) for bandwidth in (400, 4000, 40000)]
    topo_spec3 = [([
        ("/job:worker/replica:0/task:0/device:GPU:0", 1.5, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:1", 1.5, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:2", 1.5, 3<<30),
        ("/job:worker/replica:0/task:0/device:GPU:3", 1.5, 3<<30),
        ("/job:worker/replica:0/task:1/device:GPU:0", 1, 3<<30),
        ("/job:worker/replica:0/task:1/device:GPU:1", 1, 3<<30),
    ], 100, bandwidth) for bandwidth in (4000, 40000)]
    op_table = {}
    return [gen_data(gdef, prof_data, op_table, devices, inter, intra) for gdef, prof_data in models for devices, inter, intra in topo_spec1 + topo_spec2 + topo_spec3]

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
