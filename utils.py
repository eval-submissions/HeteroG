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
