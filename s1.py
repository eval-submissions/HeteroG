import tensorflow as tf
import pickle

import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

g = pickle.load(open("model.pickle", "rb"))
tf.import_graph_def(g)

server = tf.distribute.Server(tf.train.ClusterSpec({
    "tge": ["127.0.0.1:3901", "127.0.0.1:3902"]
}), job_name='tge', task_index=1, protocol="grpc")

server.join()
