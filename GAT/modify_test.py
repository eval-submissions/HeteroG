import time
import numpy as np
import tensorflow as tf

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

    def get_reward(self,strategy,index_id_dict):
        if self.strategy_reward_dict.get(str(strategy),None):
            reward= self.strategy_reward_dict.get(str(strategy))
        else:
            reward = tge.TGE(copy.deepcopy(self.gdef), self.devices).custom({index_id_dict[index]:strategy_int for index,strategy_int in enumerate(strategy)}).evaluate(self.name_cost_dict,self.folder+"/modified_strategy.json")
            #reward = np.sum(strategy*strategy)
            self.strategy_reward_dict[str(strategy)]=reward
        return np.float32(reward/(10**6))

    def directly_get_reward(self,strategy_dict):
        time = tge.TGE(copy.deepcopy(self.gdef), self.devices).custom(
            strategy_dict).evaluate(
            self.name_cost_dict,self.folder+"/best_stratey.json")
        return np.float32(time / (10 ** 6))

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
devices = (
    "/job:tge/replica:0/task:0/device:GPU:0",
    "/job:tge/replica:0/task:0/device:GPU:1",
    "/job:tge/replica:0/task:1/device:GPU:0",
    "/job:tge/replica:0/task:1/device:GPU:1"
)
env = Environment(prefix+"/graph.pbtxt",devices,prefix)
dataset = load_cora(prefix,NewWhiteSpaceTokenizer())
index_id_dict = dataset.network.get_indexer(N_TYPE_NODE).index_id_dict
feature_matrix, feature_masks = dataset.feature_matrix(bag_of_words=False, sparse=False)
nb_nodes = feature_matrix.shape[0]

changes = list()
with open("modify_test_config.json","r") as f:
    tmp = json.load(f)
    changes = tmp["changes"]

with open(prefix+"/best_time.log", "r") as f:
    txt_dict = json.load(f)
    best_time = txt_dict["time"]
    best_strategy = txt_dict["strategy"]
change_strategy = list(best_strategy.values())
for change in changes:
    old = change["old"]
    new = change["new"]
    for i in range(len(change_strategy)):
        if str(change_strategy[i])==str(old):
            change_strategy[i] = new
change_strategy = np.array(change_strategy)
print("changes",changes)
print("time before change:",env.directly_get_reward(best_strategy))
print("time after change:",env.get_reward(change_strategy,index_id_dict))
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
