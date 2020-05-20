import numpy as np
import json
import re
import sys
import tensorflow as tf
import google.protobuf.text_format as pbtf
from tensorflow.core.framework import graph_pb2

prefix=sys.argv[1]
logfile = open(prefix+"/analysis.log","w")

with open(prefix+"/best_time.log", "r") as f:
    txt_dict = json.load(f)
    best_reward = txt_dict["time"]
    best_strategy = txt_dict["strategy"]
    name_cost_dict = txt_dict["cost"]
    group =txt_dict["group"]
print("Time:",best_reward)
print("Group",group)
logfile.write("Time:{}\n".format(best_reward))
str_group = str(group)
logfile.write("Group:{}\n".format(str_group))

sorted_tuple =list()
for name, cost in name_cost_dict.items():
    sorted_tuple.append((int(cost[0]),name))

sorted_tuple.sort(reverse=True)

#strategy counter for all
counter_dict = dict()
for name,strategy in best_strategy.items():
    if counter_dict.get(str(strategy),None):
        counter_dict[str(strategy)]+=1
    else:
        counter_dict[str(strategy)]=1

for strategy,counter in counter_dict.items():
    print("Strategy:",strategy," Counter:",counter, "Ratio:",float(counter)/len(best_strategy))
    logfile.write("Strategy:{} Counter:{} Ratio:{}\n".format(strategy,counter,float(counter)/len(best_strategy)))



#top 100 operation details
#sorted_tuple = sorted_tuple[:100]
for item in sorted_tuple:
    name = item[1]
    cost = name_cost_dict[name]
    strategy = best_strategy[name]
    print("Name:",name," Strategy:",strategy," Cost:",cost)
    logfile.write("Name:{} Strategy:{} Cost:{}\n".format(name,strategy,cost))

null_gdef = graph_pb2.GraphDef()
with open(prefix+"/null_graph.pbtxt", "r")as f:
    txt = f.read()
pbtf.Parse(txt, null_gdef)
global_name_list = [nodedef.name for nodedef in null_gdef.node]

strategy_name_dict = dict()
for name,strategy in best_strategy.items():
    if strategy_name_dict.get(str(strategy),None)==None:
        strategy_name_dict[str(strategy)] = list()
    name_list = strategy_name_dict.get(str(strategy),list())
    try:
        input_node_idx = global_name_list.index(name)
    except:
        continue
    # output_node_idx = i
    input_nodedef = null_gdef.node[input_node_idx]
    if input_nodedef.op!="ApplyGradientDescent":
        continue
    size = 0
    for output_shape in input_nodedef.attr["_output_shapes"].list.shape:
        local_size = 1
        for dim in output_shape.dim:
            local_size *= np.abs(dim.size)
        size += local_size
    name_list.append((name,size,name_cost_dict.get(name,[0])[0]))

logfile.close()
import numpy as np
import matplotlib.pyplot as plt
colors = ["green","blue","red","yellow","black"]
fig = plt.figure()
ax = plt.subplot()
for i,key in enumerate(sorted(strategy_name_dict)):
    tup = strategy_name_dict[key]
    size = [float(item[1]) for item in tup]
    cost = [float(item[2]) for item in tup]

    ax.scatter(size, cost, c=colors[i%len(colors)],label=key)
plt.xlabel('size(Byte)')
plt.ylabel('cost(ms)')
plt.title(prefix)
ax.legend()
fig.savefig(prefix+"/analysis.png")


logfile = open(prefix+"/apply_gradient_analysis.log","w")
for i,key in enumerate(strategy_name_dict):
    logfile.write(key+"\n")
    tup = strategy_name_dict[key]
    for tup_item in tup:
        name = tup_item[0]
        size = tup_item[1]
        logfile.write("    name:{},size:{}Byte\n".format(name,size))
logfile.close()