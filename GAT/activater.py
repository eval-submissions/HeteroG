import numpy as np
import tensorflow as tf
import json
import os
import time
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import step_stats_pb2
import google.protobuf.text_format as pbtf
import pickle as pkl
import sys
sys.path.append('../')
from utils import write_tensorboard, setup_workers

config_dict =dict()
if os.path.exists("activate_config.txt"):
    with open("activate_config.txt", "r") as f:
        config_dict = json.load(f)


activate_graphs=config_dict.get("activate_graphs", ["data/graph1/best_graph.pbtxt","data/graph1/dp_graph.pbtxt","data/graph1/single_graph.pbtxt"])
sinks = config_dict.get("activate_sink", ["GradientDescent"])
class Activater():
    def __init__(self, activate_path, target=None, sinks=["GradientDescent"]):
        self.graph_defs = []
        self.path = activate_path
        for path in activate_path:
            gdef = graph_pb2.GraphDef()
            with open(path,"r")as f:
                txt = f.read()
            pbtf.Parse(txt,gdef)
            self.graph_defs.append(gdef)
        self.sinks = sinks
        self.target = target

    def activate(self):
        for i,graph_def in enumerate(self.graph_defs):
            tf.reset_default_graph()
            tf.import_graph_def(graph_def)
            graph = tf.get_default_graph()
            init = graph.get_operation_by_name("import/init/replica_0")
            config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True)
            #config.gpu_options.allow_growth = True
            sess = tf.Session(self.target,config=config)#, config=tf.ConfigProto(allow_soft_placement=False))
            sess.run(init)

            placeholders = (node.outputs[0] for node in graph.get_operations() if node.node_def.op == 'Placeholder')
            input_dict = { p: np.random.rand(*p.shape.as_list()) for p in placeholders }
            opt = [graph.get_operation_by_name('import/' + x) for x in self.sinks]
            for i in range(3):  #warm up
                sess.run(opt, feed_dict=input_dict)
            start_time = time.time()
            for i in range(10):
                sess.run(opt,feed_dict=input_dict)
            avg_time = (time.time()-start_time)/10
            print(self.path[i])
            print("average time:",avg_time)



workers = ["localhost:3901", "localhost:3902"]
server = setup_workers(workers, "grpc+verbs")

act = Activater(activate_graphs,server.target,sinks=sinks)
act.activate()

