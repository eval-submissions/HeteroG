# coding=utf-8
import time
import numpy as np
import tensorflow as tf
import google.protobuf.text_format as pbtf
from tensorflow.python.client import timeline
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import node_def_pb2
import sys
sys.path.append('../')
from profiler import Profiler

def model_fn():
    from tensorflow.contrib.slim.nets import vgg
    x = tf.placeholder(tf.float32, shape=(64, 224, 224, 3))
    y = tf.placeholder(tf.float32, shape=(64, 1000))
    output, _ = vgg.vgg_19(x, 1000)
    output = tf.contrib.layers.fully_connected(output, 10)
    output = tf.contrib.layers.fully_connected(output, 100000,scope="sp1")
    output = tf.contrib.layers.fully_connected(output,1000,scope="sp2")
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))
    return optimizer

def generate_edge_file(gdef):
    with open("graph.pbtxt","w") as f:
        f.write(str(gdef))
    name_list = [nodedef.name for nodedef in gdef.node]
    item_list=[]
    for i,nodedef in enumerate(gdef.node):
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
                output_node_idx = i
                input_nodedef = gdef.node[input_node_idx]
                output_shape = input_nodedef.attr["_output_shapes"].list.shape[index]
                size = 1
                for dim in output_shape.dim:
                    size*=dim.size
                item_list.append("{} {} {}".format(input_nodedef.name,nodedef.name,size))
    with open("edgelist.txt","w") as f:
        item_list = ["\n"+item if i!=0 else item for i,item in enumerate(item_list)]
        f.writelines(item_list)







def generate_feature_file(gdef):
    item_list=[]
    op_type_dict=dict()
    profiler = Profiler(gdef)
    for i,nodedef in enumerate(gdef.node):

        if not op_type_dict.get(nodedef.op,None):
            op_type_dict[nodedef.op] = len(op_type_dict.keys())
        if nodedef.op!="Placeholder":
            try:
                time = profiler.profile(nodedef.name,'/gpu:0')
            except Exception as ex:
                print(sys.stderr, 'profile error: ', ex)
                print(nodedef)
                time = 0.1
            item_list.append("{} {} {}".format(nodedef.name, time, op_type_dict[nodedef.op]))
        else:
            item_list.append("{} {} {}".format(nodedef.name, 0.1, op_type_dict[nodedef.op]))



    with open("docs.txt","w") as f:
        item_list = ["\n"+item if i!=0 else item for i,item in enumerate(item_list)]
        f.writelines(item_list)


opt = model_fn()
init = tf.global_variables_initializer()
gdef = tf.get_default_graph().as_graph_def(add_shapes=True)



generate_edge_file(gdef)
generate_feature_file(gdef)