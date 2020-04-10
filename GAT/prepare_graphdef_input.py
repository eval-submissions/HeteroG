# coding=utf-8
import time
import numpy as np
import tensorflow as tf
import google.protobuf.text_format as pbtf
from tensorflow.python.client import timeline
from tensorflow.core.framework import node_def_pb2
import sys
import json
import os
import urllib.request
import time
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import step_stats_pb2
import google.protobuf.text_format as pbtf
import pickle as pkl
sys.path.append('../')
from profiler import Profiler
from profiler import NcclProfiler
from tensorflow.distribute.cluster_resolver import TFConfigClusterResolver
import traceback

config_dict =dict()
if os.path.exists("config.txt"):
    with open("config.txt", "r") as f:
        config_dict = json.load(f)

devices=config_dict.get("devices", [
    "/job:worker/replica:0/task:0/device:GPU:0",
    "/job:worker/replica:0/task:0/device:GPU:1",
    "/job:worker/replica:0/task:1/device:GPU:0",
    "/job:worker/replica:0/task:1/device:GPU:1",
    "/job:worker/replica:0/task:2/device:GPU:0",
    "/job:worker/replica:0/task:2/device:GPU:1"

])

def setup_workers(workers, protocol="grpc"):


    param = '/'.join(server.replace(':', '%3A') for server in workers)
    for task_id, server in enumerate(workers):
        if task_id == 0: continue
        url = "http://{}:3905/{}/restart/{}/{}/{}".format(server.split(':')[0], int(time.time()) + 10, protocol, task_id, param)
        assert urllib.request.urlopen(url).read() == b'ok'
    time.sleep(1)


#workers = ["10.28.1.26:3901", "10.28.1.25:3901","10.28.1.24:3901","10.28.1.17:3901","10.28.1.16:3901"]
workers = ["10.28.1.26:3901","10.28.1.17:3901","10.28.1.16:3901"]
os.environ["TF_CONFIG"] = '{ "cluster": { "worker": ["10.28.1.26:3901","10.28.1.17:3901","10.28.1.16:3901"]  }, "task": {"type": "worker", "index": 0} }'
setup_workers(workers, "grpc")

resolver = TFConfigClusterResolver()
cluster = resolver.cluster_spec()
dist = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    tf.distribute.experimental.CollectiveCommunication.NCCL)
config = dist.update_config_proto(tf.ConfigProto())
config.ClearField("device_filters")
config.allow_soft_placement = True  # log_device_placement=True)
config.gpu_options.allow_growth = True
server = tf.distribute.Server(cluster, job_name='worker', task_index=0, protocol="grpc", config=config)


def model_fn(model_name,batch_size):
    if model_name=="vgg19":
        from tensorflow.contrib.slim.nets import vgg
        x = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
        y = tf.placeholder(tf.float32, shape=(batch_size,1000))
        output, _ = vgg.vgg_19(x, 1000)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
        optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
        return optimizer
    elif model_name=="resnet200":
        from tensorflow.contrib.slim.nets import resnet_v2
        x = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
        y = tf.placeholder(tf.float32, shape=(batch_size,1,1, 1000))
        output, _ = resnet_v2.resnet_v2_200(x, 1000)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
        optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
        return optimizer
    elif model_name=="resnet101":
        from tensorflow.contrib.slim.nets import resnet_v2
        x = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
        y = tf.placeholder(tf.float32, shape=(batch_size,1,1, 1000))
        output, _ = resnet_v2.resnet_v2_101(x, 1000)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
        optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
        return optimizer
    elif model_name=="resnet152":
        from tensorflow.contrib.slim.nets import resnet_v2
        x = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
        y = tf.placeholder(tf.float32, shape=(batch_size,1,1, 1000))
        output, _ = resnet_v2.resnet_v2_152(x, 1000)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
        optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
        return optimizer
    elif model_name=="resnet50":
        from tensorflow.contrib.slim.nets import resnet_v2
        x = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
        y = tf.placeholder(tf.float32, shape=(batch_size,1,1, 1000))
        output, _ = resnet_v2.resnet_v2_50(x, 1000)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
        optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
        return optimizer
    elif model_name=="inceptionv3":
        from tensorflow.contrib.slim.nets import inception_v3
        x = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3))
        y = tf.placeholder(tf.float32, shape=(batch_size, 1000))
        output, _ = inception_v3.inception_v3(x, 1000)
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
        optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
        return optimizer
def generate_edge_file(null_gdef,folder):
    with open(folder+"graph.pbtxt","w") as f:
        f.write(pbtf.MessageToString(null_gdef))
    name_list = [nodedef.name for nodedef in null_gdef.node]
    item_list=[]
    for i,nodedef in enumerate(null_gdef.node):
        for j,input in enumerate(nodedef.input):
            if ":" in input:
                index = int(input.split(":")[1])
                input = input.split(":")[0]
            else:
                index=0

            if input[0]=="^":
                input_node_idx = name_list.index(input[1:])
                output_node_idx = i
                item_list.append("{} {} {}".format(input[1:], nodedef.name, 1))
            else:
                input_node_idx = name_list.index(input)
                #output_node_idx = i
                input_nodedef = null_gdef.node[input_node_idx]
                #output_shape = input_nodedef.attr["_output_shapes"].list.shape[index]
                #size = 1
                #for dim in output_shape.dim:
                #    size*=dim.size
                item_list.append("{} {} {}".format(input_nodedef.name,nodedef.name,1))
    with open(folder+"edgelist.txt","w") as f:
        item_list = ["\n"+item if i!=0 else item for i,item in enumerate(item_list)]
        f.writelines(item_list)

def generate_nccl_model():
    model = NcclProfiler(devices,server.target).profile()
    with open("data/nccl_model.pkl","wb") as f:
        pkl.dump(model,f)


def generate_feature_file(folder,index):
    batch_size=48
    final_dict=dict()
    opt = model_fn(models[index],None)
    init = tf.global_variables_initializer()
    null_gdef = tf.get_default_graph().as_graph_def(add_shapes=True)
    with open(folder + "null_graph.pbtxt", "w") as f:
        f.write(pbtf.MessageToString(null_gdef))
    tf.reset_default_graph()

    generate_edge_file(null_gdef,folder)
    if os.path.exists("op_type_dict.json"):
        with open("op_type_dict.json", "r") as f:
            op_type_dict=json.load(f)
    else:
        op_type_dict = dict()
    replica_num = [1,2,3,4,6,8,12]
    item_list=[]
    times_dict=dict()
    for replica_times in range(len(replica_num)):
        tf.reset_default_graph()
        run_metadata = None
        run_meta_file_name = "run_metadata"+str(int(batch_size/replica_num[replica_times]))+".pbtxt"
        if os.path.exists(folder+run_meta_file_name):
            run_metadata = tf.RunMetadata()
            with open(folder+run_meta_file_name, "r")as f:
                txt = f.read()
            pbtf.Parse(txt, run_metadata)
        else:
            #opt = model_fn(models[index],batch_size/replica_num[replica_times])
            #init = tf.global_variables_initializer()
            #gdef = tf.get_default_graph().as_graph_def(add_shapes=True)
            profiler = Profiler(null_gdef,int(batch_size/replica_num[replica_times]),server.target)
        for i,nodedef in enumerate(null_gdef.node):
            times = times_dict.get(nodedef.name,'')
            if op_type_dict.get(nodedef.op,-1)==-1:
                op_type_dict[nodedef.op] = len(op_type_dict.keys())

            for i in range(len(devices)):
                try:
                    time = profiler.profile(nodedef.name,devices[i],run_metadata)
                except Exception as ex:
                    print(sys.stderr, 'profile error: ', ex)
                    print(nodedef)
                    traceback.print_exc()
                    time = 0
                new_time = time
                item=final_dict.get((nodedef.name,replica_num[replica_times]),None)
                if item==None:
                    final_dict[(nodedef.name,replica_num[replica_times])]=list()
                    item = final_dict[(nodedef.name,replica_num[replica_times])]
                item.append(new_time)
                times+=str(new_time)+" "
            times_dict[nodedef.name] = times
    name_list = [nodedef.name for nodedef in null_gdef.node]
    for i, nodedef in enumerate(null_gdef.node):
        size=0
        for j,input in enumerate(nodedef.input):
            if ":" in input:
                index = int(input.split(":")[1])
                input = input.split(":")[0]
            else:
                index=0

            if input[0]=="^":
                continue
            else:
                input_node_idx = name_list.index(input)
                #output_node_idx = i
                input_nodedef = null_gdef.node[input_node_idx]
                output_shape = input_nodedef.attr["_output_shapes"].list.shape[index]
                local_size=1
                for dim in output_shape.dim:
                    local_size*=dim.size
                size+=local_size
        times = times_dict[nodedef.name]
        item_list.append("{} {} {}{} {}".format(nodedef.name, op_type_dict[nodedef.op],times,size,batch_size))
    for i,nodedef in enumerate(null_gdef.node):
        if nodedef.name not in name_list:
            item_list.append("{} {} {}{} {}".format(nodedef.name, op_type_dict[nodedef.op], 0, 0, batch_size))

    with open(folder+"docs.txt","w") as f:
        item_list = ["\n"+item if i!=0 else item for i,item in enumerate(item_list)]
        f.writelines(item_list)
    with open("op_type_dict.json", "w") as f:
        json.dump(op_type_dict,f)
    with open(folder+"cost.pkl", "wb") as f:
        pkl.dump(final_dict,f)

models = ["vgg19","resnet200","resnet50","resnet101","resnet152","inceptionv3"]#,"bert"]
for i in range(len(models)):
    tf.reset_default_graph()
    folder = "data/graph"+str(i+1)+"/"
    generate_feature_file(folder,i)
#generate_nccl_model()