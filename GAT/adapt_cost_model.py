import pickle as pkl
import sys
import multiprocessing as mp
sys.path.append('../')
from utils import adapt_batchsize
from profiler import Profiler
import tensorflow as tf
import os
import json




config_dict = dict()
if os.path.exists("config.txt"):
    with open("config.txt", "r") as f:
        config_dict = json.load(f)

batch_sizes = config_dict.get("batch_sizes", [48 * 2, 288 * 2, 6 * 2])





def adapt_cost(folder,init_batch,after_batch,max_replica):
    with open(folder+"cost.pkl","rb") as f:
        name_cost_dict = pkl.load(f)
    name_cost_dict = adapt_batchsize(name_cost_dict,init_batch,after_batch,max_replica)
    with open(folder+"new_cost.pkl","wb") as f:
        pkl.dump(name_cost_dict,f)



models = ["vgg19","resnet200","resnet50","resnet101","resnet152","inceptionv3","transformer","bert"]
processes = []
for i in range(len(models)):
    if i==6:
        tf.reset_default_graph()
        folder = "data/graph"+str(i+1)+"/"
        #adapt_cost(folder,288,288*3,18)
        processes.append(mp.Process(target=adapt_cost,args=(folder,288,batch_sizes[1],8,)))
    if i==7:
        tf.reset_default_graph()
        folder = "data/graph"+str(i+1)+"/"
        #adapt_cost(folder,12,12*3,18)
        processes.append(mp.Process(target=adapt_cost,args=(folder,6,batch_sizes[2],8,)))

    else:
        tf.reset_default_graph()
        folder = "data/graph"+str(i+1)+"/"
        #adapt_cost(folder,36,36*3,18)
        processes.append(mp.Process(target=adapt_cost,args=(folder,24,batch_sizes[0],8,)))

for process in processes:
    process.start()
for process in processes:
    process.join()
