import tensorflow as tf

def get_device_list():
    from tensorflow.python.client import device_lib
    return [x.name for x in device_lib.list_local_devices()]

def write_tensorboard(graph):
    writer = tf.summary.FileWriter('.')
    writer.add_graph(graph)
    writer.flush()
