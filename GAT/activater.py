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
from tensorflow.distribute.cluster_resolver import TFConfigClusterResolver

sys.path.append('../')

config_dict =dict()
if os.path.exists("activate_config.txt"):
    with open("activate_config.txt", "r") as f:
        config_dict = json.load(f)

def setup_workers(workers, protocol="grpc"):
    import urllib.request
    import time

    param = '/'.join(server.replace(':', '%3A') for server in workers)
    for task_id, server in enumerate(workers):
        if task_id == 0: continue
        url = "http://{}:3905/{}/restart/{}/{}/{}".format(server.split(':')[0], int(time.time()) + 10, protocol, task_id, param)
        assert urllib.request.urlopen(url).read() == b'ok'
    time.sleep(1)
activate_graphs=config_dict.get("activate_graphs", ["data/graph1/nccl_dp_graph.pbtxt","data/graph1/grpc_dp_graph.pbtxt","data/graph1/single_graph.pbtxt","data/graph1/best_graph.pbtxt"])
sinks = config_dict.get("activate_sink", ["GradientDescent"])
class Activater():
    def __init__(self, activate_path, sinks=["GradientDescent"]):
        self.graph_defs = []
        self.path = activate_path
        for path in activate_path:
            gdef = graph_pb2.GraphDef()
            with open(path,"r")as f:
                txt = f.read()
            pbtf.Parse(txt,gdef)
            self.graph_defs.append(gdef)

        resolver = TFConfigClusterResolver()
        cluster = resolver.cluster_spec()
        dist = tf.distribute.experimental.MultiWorkerMirroredStrategy(
            tf.distribute.experimental.CollectiveCommunication.NCCL)
        self.config = dist.update_config_proto(tf.ConfigProto())
        self.config.ClearField("device_filters")
        self.config.allow_soft_placement = True  # log_device_placement=True)
        self.config.gpu_options.allow_growth = True
        self.server = tf.distribute.Server(cluster, job_name='worker', task_index=0, protocol="grpc", config=self.config)

        self.sinks = sinks
        self.target = self.server.target

    def activate(self,batch_size):
        for k,graph_def in enumerate(self.graph_defs):
            tf.reset_default_graph()


            tf.import_graph_def(graph_def)
            graph = tf.get_default_graph()
            init = graph.get_operation_by_name("import/init/replica_0")
            sess = tf.Session(self.target,config=self.config)#, config=tf.ConfigProto(allow_soft_placement=False))
            sess.run(init)
            input_dict = None
            '''
            placeholders = [node.outputs[0] for node in graph.get_operations() if node.node_def.op == 'Placeholder']
            shapes = [(p.shape.as_list()) for p in placeholders ]
            for shape in shapes:
                shape[0]=batch_size
            input_dict = { p: np.random.rand(*shapes[i]) for i,p in enumerate(placeholders) }
            '''
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

os.environ["TF_CONFIG"] = '{ "cluster": { "worker": ["localhost:3901", "localhost:3902"] }, "task": {"type": "worker", "index": 0} }'

setup_workers(workers, "grpc")

act = Activater(activate_graphs,sinks=sinks)
act.activate(48)

