import time
import numpy as np
import tensorflow as tf
import os

from models import GAT
from utils import process
from data_process.dataset import GraphDataset, WhiteSpaceTokenizer,NewWhiteSpaceTokenizer
from data_process.example import load_M10, load_cora, load_dblp
from data_process.meta_network import MetaNetwork, N_TYPE_NODE, N_TYPE_LABEL, IdIndexer
import google.protobuf.text_format as pbtf
from tensorflow.core.framework import graph_pb2
import copy
import sys
import json
sys.path.append('../')
import tge
prefix=sys.argv[1]

config_dict =dict()
if os.path.exists("config.txt"):
    with open("config.txt", "r") as f:
        config_dict = json.load(f)
devices=config_dict.get("devices", [
    "/job:tge/replica:0/task:0/device:GPU:0",
    "/job:tge/replica:0/task:0/device:GPU:1",
    "/job:tge/replica:0/task:1/device:GPU:0",
    "/job:tge/replica:0/task:1/device:GPU:1"
])
class Environment(object):
    def __init__(self,gdef_path,devices,folder):

        self.gdef = graph_pb2.GraphDef()
        with open(gdef_path,"r")as f:
            txt = f.read()
        pbtf.Parse(txt,self.gdef)
        self.folder = folder
        self.strategy_reward_dict=dict()
        self.name_cost_dict = self.get_name_cost_dict()
        self.devices =devices
        self._tge = tge.TGE(self.gdef, devices)

    def get_reward(self,strategy,index_id_dict,trace=""):
        if self.strategy_reward_dict.get(str(strategy),None):
            reward= self.strategy_reward_dict.get(str(strategy))
        else:
            bandwidth = config_dict.get("bandwidth",None)
            if bandwidth==None:
                intra = "5000"
                inter = "1250"
            else:
                intra = bandwidth[0]
                inter = bandwidth[1]
            reward = tge.TGE(copy.deepcopy(self.gdef), self.devices).custom({index_id_dict[index]:strategy_int for index,strategy_int in enumerate(strategy)}).set_bandwidth(intra,inter).evaluate(self.name_cost_dict,trace)
            #reward = np.sum(strategy*strategy)
            self.strategy_reward_dict[str(strategy)]=reward
        return np.float32(reward/(10**6))

    def get_name_cost_dict(self):
        name_cost_dict = dict()
        with open(self.folder+"/docs.txt", "r") as f:
            for line in f.readlines():
                line = line.strip()
                items = line.split(" ")
                name = items[0]
                cost = list(np.array(items[-len(devices):]))
                name_cost_dict[name] = cost
        return name_cost_dict

env = Environment(prefix+"/graph.pbtxt",devices,prefix)
dataset = load_cora(prefix,NewWhiteSpaceTokenizer())
index_id_dict = dataset.network.get_indexer(N_TYPE_NODE).index_id_dict
feature_matrix, feature_masks = dataset.feature_matrix(bag_of_words=False, sparse=False)
nb_nodes = feature_matrix.shape[0]

with open("test_config.json","r") as f:
    tmp = json.load(f)
    strategies = tmp["strategies"]
for _strategy in strategies:
    strategy = list()
    for i in range(nb_nodes):
        strategy.append(_strategy)
    arr_strategy = np.array(strategy)
    print("strategy:",_strategy)
    print(env.get_reward(arr_strategy,index_id_dict,prefix+"/"+str(_strategy)+".json"))
'''
name_cost_dict = env.get_name_cost_dict()
cost = list(name_cost_dict.values())
cost.sort()
for name in name_cost_dict.keys():
    if "Back" in name:
        print(name,name_cost_dict[name])
    if name_cost_dict[name]>cost[-100]:
        print(name,name_cost_dict[name])
'''     
