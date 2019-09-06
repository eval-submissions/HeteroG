import tensorflow as tf
import urllib.request
import time

def get_device_list():
    from tensorflow.python.client import device_lib
    return [x.name for x in device_lib.list_local_devices()]

def write_tensorboard(graph):
    writer = tf.summary.FileWriter('.')
    writer.add_graph(graph)
    writer.flush()

def restart_workers(workers):
    param = '/'.join(server.replace(':', '%3A') for server in workers)
    for task_id, server in enumerate(workers):
        if task_id == 0:
            continue
        url = "http://{}:3905/{}/restart/{}/{}".format(server.split(':')[0], int(time.time()) + 10, task_id, param)
        assert urllib.request.urlopen(url).read() == b'ok'
    time.sleep(1)
