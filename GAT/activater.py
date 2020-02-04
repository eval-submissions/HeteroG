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
from tensorflow.python.client import timeline

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

    def activate(self,batch_size):
        for k,graph_def in enumerate(self.graph_defs):
            tf.reset_default_graph()
            tf.import_graph_def(graph_def)
            graph = tf.get_default_graph()
            init = graph.get_operation_by_name("import/init/replica_0")
            config = tf.ConfigProto(allow_soft_placement=True)  # log_device_placement=True)
            #config.gpu_options.allow_growth = True
            sess = tf.Session(self.target,config=config)#, config=tf.ConfigProto(allow_soft_placement=False))
            sess.run(init)

            placeholders = [node.outputs[0] for node in graph.get_operations() if node.node_def.op == 'Placeholder']
            shapes = [(p.shape.as_list()) for p in placeholders ]
            for shape in shapes:
                shape[0]=batch_size
            input_dict = { p: np.random.rand(*shapes[i]) for i,p in enumerate(placeholders) }
            opt=[]
            for sink in self.sinks:
                for i in range(10):
                    try:
                        op=graph.get_operation_by_name('import/' + sink+"/replica_"+str(i))
                        opt.append(op)
                    except:
                        break
            #opt = [graph.get_operation_by_name('import/' + x) for x in self.sinks]
            for j in range(3):  #warm up
                sess.run(opt, feed_dict=input_dict)
            start_time = time.time()
            for j in range(10):
                sess.run(opt,feed_dict=input_dict)
            avg_time = (time.time()-start_time)/10
            print(self.path[k])
            print("average time:",avg_time)

            run_meta = tf.RunMetadata()
            run_opt = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
            sess.run(opt, feed_dict=input_dict,
                     options=run_opt,
                     run_metadata=run_meta
                     )
            tl = timeline.Timeline(run_meta.step_stats)
            with open(self.path[k].split(".")[0]+"_timeline.json", "w") as fo:
                fo.write(tl.generate_chrome_trace_format())



workers = ["localhost:3901", "localhost:3902"]
server = setup_workers(workers, "grpc")

act = Activater(activate_graphs,server.target,sinks=sinks)
act.activate(48)

