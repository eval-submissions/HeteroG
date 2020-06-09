def get_device_list():
    from tensorflow.python.client import device_lib
    return [x.name for x in device_lib.list_local_devices()]

def write_tensorboard(graph):
    import tensorflow as tf
    writer = tf.summary.FileWriter('.')
    writer.add_graph(graph)
    writer.flush()

def setup_workers(workers, protocol="grpc"):
    import tensorflow as tf
    import urllib.request
    import time

    param = '/'.join(server.replace(':', '%3A') for server in workers)
    for task_id, server in enumerate(workers):
        if task_id == 0: continue
        url = "http://{}:3905/{}/restart/{}/{}/{}".format(server.split(':')[0], int(time.time()) + 10, protocol, task_id, param)
        assert urllib.request.urlopen(url).read() == b'ok'
    time.sleep(1)

    return tf.distribute.Server(tf.train.ClusterSpec({
        "tge": workers
    }), job_name='tge', task_index=0, protocol=protocol)

def op_def_dict():
    import tensorflow.core.framework as tfpb
    from google.protobuf import text_format

    oplist = tfpb.op_def_pb2.OpList()
    text_format.Parse(open("ops.pbtxt").read(), oplist)

    return { op.name: op for op in oplist.op }

def adapt_batchsize(profile_data, original_batchsize, new_batchsize, nrep_limit):
    from sklearn.linear_model import LinearRegression
    def linear_pred(data, x):
        try:
            return [y for _x, y in data if _x == x ][0]
        except:
            model = LinearRegression().fit([[x] for x, y in data], [[y] for x, y in data])
            return max(int(model.predict([[x]])[0][0]), 0)

    nodes = set((node for node, nrep in profile_data.keys()))
    ndev = len(profile_data.values().__iter__().__next__())

    data_points = { (node, dev): [(original_batchsize // nrep, cost_list[dev]) for ((_name, nrep), cost_list) in profile_data.items() if _name == node ] for node in nodes for dev in range(ndev) }
    return { (node, nrep): [linear_pred(data_points[(node, dev)], new_batchsize // nrep) for dev in range(ndev)] for node in nodes for nrep in range(1, nrep_limit+1) if new_batchsize % nrep == 0 }

def group_around_topk_costs(gdef, groups, profile_data, k):
    # cores are the largest cost node in each group, which are used to represent the whole group. Centers are topk cores.
    def cost(x):
        return profile_data[(x, 1)][0]

    groups = { gdef.node[i].name: id for i, id in enumerate(groups) } # node -> group
    cores = {} # group -> core node
    for node, group in groups.items():
        if group not in cores or cost(cores[group]) < cost(node):
            cores[group] = node
    import heapq
    centers = heapq.nlargest(k, cores.values(), key=cost)

    import networkx as nx
    G = nx.Graph()
    for node in gdef.node:
        name = node.name
        G.add_node(name)
        for input in node.input:
            if input[0] == '^':
                x = input[1:]
            else:
                x = input.split(':')[0]
            G.add_edge(name, x)

    for center in centers:
        for node, distance in nx.single_source_shortest_path_length(G, center).items():
            G.nodes[node][center] = distance

    new_groups = {} # group -> new_group
    for group, core in cores.items():
        if core in centers:
            new_groups[group] = group
            continue
        center = min(G.nodes[core], key=G.nodes[core].get, default=None)
        new_groups[group] = groups.get(center, 0)

    id = 0
    new_id = {} # old_id -> new_id
    result = []
    for node in gdef.node:
        old_group = groups[node.name]
        new_group = new_groups[old_group]
        if new_group not in new_id:
            new_id[new_group] = id
            id += 1
        result.append(new_id[new_group])

    return result

def save(var, file):
    import pickle
    with open(file, 'wb') as f:
        pickle.dump(var, f)

def load(file):
    import pickle
    with open(file, 'rb') as f:
        return pickle.load(f)
