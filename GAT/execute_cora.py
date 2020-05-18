import time
import numpy as np
import tensorflow as tf
from scipy.sparse import csr_matrix
from models import GAT
from models import SpGAT
from gat_utils import process
from data_process.dataset import GraphDataset, WhiteSpaceTokenizer,NewWhiteSpaceTokenizer
from data_process.example import load_M10, load_cora, load_dblp
from data_process.meta_network import MetaNetwork, N_TYPE_NODE, N_TYPE_LABEL, IdIndexer
from transformer.model import transformer
import google.protobuf.text_format as pbtf
from tensorflow.core.framework import graph_pb2
from sklearn.preprocessing import StandardScaler
import copy
import sys
import os
import scipy.sparse as sp
import traceback
import pickle
sys.path.append('../')
import tge
import json
import pickle as pkl
import multiprocessing as mp
from utils import group_around_topk_costs
import logging
import math
def InitLog():
    log = logging.getLogger(__name__)
    log.setLevel(logging.DEBUG)
    # log to txt
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler = logging.FileHandler("log/log_%s.txt" % time.strftime("%Y-%m-%d-%H-%M-%S"))
    # handler = logging.handlers.RotatingFileHandler("log_%s.txt" % time.strftime("%Y-%m-%d %H-%M-%S"),maxBytes=1024*1024,backupCount=50)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    # log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    log.addHandler(handler)
    log.addHandler(console)
    return log

logger = InitLog()
variable_ops=["Variable", "VariableV2", "AutoReloadVariable",
                   "MutableHashTable", "MutableHashTableV2",
                   "MutableHashTableOfTensors", "MutableHashTableOfTensorsV2",
                   "MutableDenseHashTable", "MutableDenseHashTableV2",
                   "VarHandleOp", "BoostedTreesEnsembleResourceHandleOp",
                   "BoostedTreesQuantileStreamResourceHandleOp",
                   "ResourceConditionalAccumulator",
                   "DecisionTreeResource"]


checkpt_file = 'pre_trained/cora/mod_cora.ckpt'
_dataset = 'cora'

config_dict = dict()
if os.path.exists("config.txt"):
    with open("config.txt", "r") as f:
        config_dict = json.load(f)

# training params
os.environ["CUDA_VISIBLE_DEVICES"]=config_dict.get("CUDA_VISIBLE_DEVICES","0,1")
#os.environ["TF_XLA_FLAGS"]="--tf_xla_cpu_global_jit"
batch_size = 1
nb_epochs = 100000
patience = 100
lr = config_dict.get("learning_rate", 0.01)  # learning rate
l2_coef = 0.0002  # weight decay
hid_units = [512]  # numbers of hidden units per each attention head in each layer
n_heads = [4, 4]  # additional entry for the output layer
place_hid_units = [1024, 256]
place_n_heads = [4,4,1]
residual = False


global_batch_size=288

n_layer=12
n_head=8
d_head=64
d_inner=2048
group_num = 2000
bsz =1




nonlinearity = tf.nn.elu
model = SpGAT
is_transformer = True
print('Dataset: ' + _dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))
feature_folders = config_dict.get("inputs",["data/graph1", "data/graph2", "data/graph3", "data/graph4", "data/graph5", "data/graph6","data/graph7","data/graph8"])
sinks =  config_dict.get("sinks",[["GradientDescent"], ["GradientDescent"], ["GradientDescent"], ["GradientDescent"], ["GradientDescent"], ["GradientDescent"],["GradientDescent"],["GradientDescent"]])
sample_times = 2
devices = config_dict.get("devices", [
    "/job:worker/replica:0/task:0/device:GPU:0",
    "/job:worker/replica:0/task:0/device:GPU:1",
    "/job:worker/replica:0/task:1/device:GPU:0",
    "/job:worker/replica:0/task:1/device:GPU:1",
    "/job:worker/replica:0/task:2/device:GPU:0",
    "/job:worker/replica:0/task:2/device:GPU:1"

])
batch_sizes = config_dict.get("batch_sizes", [48 * 2, 288 * 2, 6 * 2])

max_replica_num = config_dict.get("max_replica_num", len(devices))
show_interval = 3
device_mems = config_dict.get("device_mems", [16 * 10e9, 16 * 10e9, 16 * 10e9, 16 * 10e9])

sample_prob = 0.1

d_model= 512

def post_process_device_choice(device_choice,batch_size):
    def post_func1(item):
        item1 = list(item[:len(item) - 1])
        batch_size = item[-1]
        if sum(item1) == 0:
            item1[0] = 1
            return item1
        while sum(item1) > batch_size:
            index = item1.index(max(item1))
            item1[index] -= 1
        while batch_size % sum(item1):
            index = item1.index(max(item1))
            item1[index] -= 1
        return np.array(item1)
    #post process and align to batch size
    new_batch_size=np.ones(shape=(device_choice.shape[0],1)) * batch_size
    device_choice=np.array(list(map(post_func1,np.concatenate((device_choice,new_batch_size),axis=1))),dtype=np.int32)

    replica_mask = np.zeros(shape=(device_choice.shape[0],device_choice.shape[1]*(max_replica_num+1)),dtype=np.int32)
    for i,item in enumerate(device_choice):
        for j,num in enumerate(item):
            replica_mask[i][j*(max_replica_num+1)+num]=1
    return device_choice,replica_mask


class strategy_pool(object):
    def __init__(self,folder_path,node_num,index_id_dict,env,batch_size,init_group,sink):
        self.folder_path = folder_path
        self.node_num = node_num
        self.index_id_dict = index_id_dict
        self.env = env
        self.sink = sink
        self.init_group = init_group
        self.init_group_num = max(self.init_group)+1
        if os.path.exists(self.folder_path+"/pool.pkl"):
            with open(self.folder_path+"/pool.pkl","rb") as f:
                self.strategies= mp.Manager().list(pkl.load(f))
                for j, strategy in enumerate(self.strategies):
                    group = strategy["group"]
                    if len(group)!=self.init_group_num:
                        self.strategies.pop(j)
            self.save_strategy_pool()
        else:
            self.strategies = mp.Manager().list()

        self.rewards = [item["reward"] for item in self.strategies] if len(self.strategies) else [-sys.maxsize]
        self.batch_size = batch_size

        # even data parallel 1
        #device_choice = np.zeros(shape=(self.node_num, len(devices)), dtype=np.int32)
        if True:
            group = np.array(self.init_group)
            device_choice = np.ones(shape=(self.init_group_num, len(devices)), dtype=np.int32)*2

            device_choice,replica_mask = post_process_device_choice(device_choice,self.batch_size)
            ps_or_reduce = np.ones(shape=(self.init_group_num, ), dtype=np.int32)
            reward,out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict,self.sink,group,record=True,record_name="full_nccl_dp_graph.pbtxt",record_best=False,from_strategy_pool=True)
            if not out_of_memory:
                self.insert(reward, device_choice, replica_mask,ps_or_reduce,group,force_insert=True)


            group = np.array(self.init_group)
            device_choice = np.ones(shape=(self.init_group_num, len(devices)), dtype=np.int32)
            for item in device_choice:
                item[0]=2
                item[1]=2
            device_choice,replica_mask = post_process_device_choice(device_choice,self.batch_size)
            ps_or_reduce = np.ones(shape=(self.init_group_num, ), dtype=np.int32)
            reward,out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict,self.sink,group,record=True,record_name="partial_nccl_dp_graph.pbtxt",record_best=False,from_strategy_pool=True)
            if not out_of_memory:
                self.insert(reward, device_choice, replica_mask,ps_or_reduce,group,force_insert=True)


            # even data parallel 2
            #device_choice = np.zeros(shape=(self.node_num, len(devices)), dtype=np.int32)

            group = np.array(self.init_group)
            device_choice = np.ones(shape=(self.init_group_num, len(devices)), dtype=np.int32)

            device_choice,replica_mask = post_process_device_choice(device_choice,self.batch_size)
            ps_or_reduce = np.ones(shape=(self.init_group_num, ), dtype=np.int32)
            reward,out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict,self.sink,group,record=True,record_name="nccl_dp_graph.pbtxt",record_best=False,from_strategy_pool=True)
            if not out_of_memory:
                self.insert(reward, device_choice, replica_mask,ps_or_reduce,group,force_insert=True)

        #    self.insert(reward, device_choice, replica_mask,ps_or_reduce,group)

            group = np.array(self.init_group)
            device_choice = np.ones(shape=(self.init_group_num, len(devices)), dtype=np.int32)

            device_choice, replica_mask = post_process_device_choice(device_choice, self.batch_size)
            ps_or_reduce = np.zeros(shape=(self.init_group_num,), dtype=np.int32)
            reward, out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict, self.sink, group,
                                                         record=True, record_name="grpc_dp_graph.pbtxt", record_best=False,
                                                         from_strategy_pool=True)
            if not out_of_memory:
                self.insert(reward, device_choice, replica_mask,ps_or_reduce,group,force_insert=True)

            #    self.insert(reward, device_choice, replica_mask,ps_or_reduce,group)


            group = np.array(self.init_group)
            device_choice = np.array([np.arange(len(devices))%1 for i in range(self.init_group_num)])

            device_choice, replica_mask = post_process_device_choice(device_choice, self.batch_size)
            ps_or_reduce = np.ones(shape=(self.init_group_num,), dtype=np.int32)
            reward, out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict, self.sink, group,
                                                         record=True, record_name="single_graph.pbtxt",
                                                         record_best=False, from_strategy_pool=True)
            if not out_of_memory:
                self.insert(reward, device_choice, replica_mask,ps_or_reduce,group,force_insert=True)

            group = np.array(self.init_group)
            device_choice = np.array([np.arange(len(devices))%1 for i in range(self.init_group_num)])
            for i,item in enumerate(device_choice):
                item[i%len(devices)]=1

            device_choice, replica_mask = post_process_device_choice(device_choice, self.batch_size)
            ps_or_reduce = np.ones(shape=(self.init_group_num,), dtype=np.int32)
            reward, out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict, self.sink, group,
                                                         record=True, record_name="model_parallel_graph.pbtxt",
                                                         record_best=False, from_strategy_pool=True)
            if not out_of_memory:
                self.insert(reward, device_choice, replica_mask,ps_or_reduce,group,force_insert=True)


            group = np.array(self.init_group)
            device_choice = np.array([np.arange(len(devices))%1 for i in range(self.init_group_num)])
            for i,item in enumerate(device_choice):
                item[i//math.ceil(len(device_choice)/len(devices))]=1

            device_choice, replica_mask = post_process_device_choice(device_choice, self.batch_size)
            ps_or_reduce = np.ones(shape=(self.init_group_num,), dtype=np.int32)
            reward, out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict, self.sink, group,
                                                         record=True, record_name="model_parallel2_graph.pbtxt",
                                                         record_best=False, from_strategy_pool=True)
            if not out_of_memory:
                self.insert(reward, device_choice, replica_mask,ps_or_reduce,group,force_insert=True)


    def get_length(self):
        return len(self.strategies)

    def get_stratey_list(self,device_choice,ps_or_reduce):
        new_device_array = device_choice
        ps_or_reduce = np.reshape(ps_or_reduce, (ps_or_reduce.shape[0], 1))
        new_device_array = np.concatenate((ps_or_reduce,new_device_array),axis=1)
        return new_device_array.tolist()

    def save_strategy_pool(self):
        if len(self.strategies)==0:
            return
        with open(self.folder_path + "/pool.pkl", "wb") as f:
            pkl.dump(list(self.strategies),f)
        with open(self.folder_path + "/pool.log", "w") as f:
            f.write(str([item["reward"] for item in self.strategies]))
            for i in range(4):
                f.write("\nstrategy"+str(i)+":\n")
                f.write(str(self.strategies[np.random.randint(len(self.strategies))]["strategy_list"]))


    def insert(self,reward,device_choice,replica_mask,ps_or_reduce,group,force_insert=False):
        def comp_fc(item):
            item1 = item[:int(len(item) / 2)]
            item2 = item[int(len(item) / 2):]
            return 0 if all(item1 == item2) else 1
        strategy_list = self.get_stratey_list(device_choice, ps_or_reduce)
        if force_insert:
            self.strategies.append({"replica_mask": replica_mask, "strategy_list": strategy_list, "reward": reward,
                                    "device_choice": device_choice, "ps_or_reduce": ps_or_reduce,"group":group})

            self.save_strategy_pool()
            self.rewards.append(reward)
            return

        if len(self.strategies)<20 and reward>np.mean(self.rewards):
            for j,strategy in enumerate(self.strategies):
                exist_device_choice = (strategy["device_choice"])
                if len(exist_device_choice)!=len(device_choice):
                    continue
                diff_list = list(map(comp_fc,np.concatenate((device_choice,exist_device_choice),axis=1)))
                if sum(diff_list)/len(diff_list)<0.2:
                    if reward>strategy["reward"]:
                        self.strategies.append({"replica_mask":replica_mask,"strategy_list":strategy_list,"reward":reward,"device_choice":device_choice,"ps_or_reduce":ps_or_reduce,"group":group})
                        self.strategies.pop(j)
                        self.save_strategy_pool()
                        self.rewards = [item["reward"] for item in self.strategies]
                    return
            self.strategies.append({"replica_mask": replica_mask, "strategy_list": strategy_list, "reward": reward,
                                    "device_choice": device_choice, "ps_or_reduce": ps_or_reduce,"group":group})

            self.save_strategy_pool()
            self.rewards.append(reward)
        elif len(self.strategies)>=10 and reward>np.mean(self.rewards):
            for j,strategy in enumerate(self.strategies):
                exist_device_choice = (strategy["device_choice"])
                if len(exist_device_choice)!=len(device_choice):
                    continue
                diff_list = list(map(comp_fc,np.concatenate((device_choice,exist_device_choice),axis=1)))
                if sum(diff_list)/len(diff_list)<0.2:
                    if reward>strategy["reward"]:
                        self.strategies.append({"replica_mask":replica_mask,"strategy_list":strategy_list,"reward":reward,"device_choice":device_choice,"ps_or_reduce":ps_or_reduce,"group":group})
                        self.strategies.pop(j)
                        self.save_strategy_pool()
                        self.rewards = [item["reward"] for item in self.strategies]
                    return
            index = self.rewards.index(min(self.rewards))
            self.strategies.pop(index)
            self.strategies.append({"replica_mask": replica_mask, "strategy_list": strategy_list, "reward": reward,
                                    "device_choice": device_choice, "ps_or_reduce": ps_or_reduce,"group":group})

            self.save_strategy_pool()
            self.rewards = [item["reward"] for item in self.strategies]

    def choose_strategy(self):
        if len(self.strategies)==0:
            return None
        self.rewards = [item["reward"] for item in self.strategies]
        index = np.random.randint(0,len(self.strategies))
        #index = self.rewards.index(max(self.rewards))
        return self.strategies[index]
def reward_func(item):
    new_device_array = np.zeros(shape=(len(devices)),dtype=np.int32)
    for j in range(len(item)):
        if item[j]!=-1 and item[j]!=len(devices):
            new_device_array[item[j]]+=1
    return new_device_array
class Environment(object):
    def __init__(self,gdef_path,devices,folder_path,batch_size,init_group,sink):

        self.gdef = graph_pb2.GraphDef()
        with open(gdef_path,"r")as f:
            txt = f.read()
        pbtf.Parse(txt,self.gdef)
        self.folder_path = folder_path
        self.random_strategy=list()
        self.best_strategy = mp.Manager().dict()
        self.best_strategy["time"] = sys.maxsize
        self.batch_size = batch_size
        self.devices =devices
        self.sink =sink
        self.init_group = init_group
        with open("nccl_model.pkl","rb") as f:
            self.nccl_model=pkl.load(f)

        bandwidth = config_dict.get("bandwidth",None)
        if bandwidth==None:
            self.intra = "5000"
            self.inter = "1250"
        else:
            self.intra = bandwidth[0]
            self.inter = bandwidth[1]

        self.null_gdef = graph_pb2.GraphDef()
        with open(folder_path+"/null_graph.pbtxt","r")as f:
            txt = f.read()
        pbtf.Parse(txt,self.null_gdef)
        self.name_cost_dict = self.get_name_cost_dict()
        if os.path.exists(folder_path+"/best_time.log"):
            with open(folder_path+"/best_time.log", "r") as f:
                tmp = json.load(f)
                for key,value in tmp.items():
                    self.best_strategy[key] = value
            with open(self.folder_path+"/best_time.log", "w") as f:

                cost_dict=dict()
                for key, value in self.name_cost_dict.items():
                    name = key[0]
                    replica_num=key[1]
                    if replica_num==1:
                        cost_dict[name] = value
                self.best_strategy["cost"] = cost_dict
                json.dump(self.best_strategy.copy(), f)
            _tge = tge.TGE(copy.deepcopy(self.null_gdef), self.devices, sink)
            time_mem_tuple = _tge.custom(self.best_strategy["strategy"]).fill_batchsize(self.batch_size).set_nccl_model(self.nccl_model).use_collective().set_bandwidth(self.intra, self.inter).evaluate(self.name_cost_dict,self.folder_path+"/best_graph.json")

            best_graph_def =tge.TGE(copy.deepcopy(self.null_gdef), self.devices, self.sink).custom(self.best_strategy["strategy"]).replace_placeholder(batch_size).use_collective().compile().get_result()
            with open(self.folder_path+"/best_graph.pbtxt", "w") as f:
                f.write(str(best_graph_def))





    def get_reward2(self,device_choice,ps_or_reduce,index_id_dict,sink,group,record=False,record_name=None,record_best=True,from_strategy_pool=False):
        out_of_memory=False
        #new_device_array = np.zeros(shape=(device_choice.shape[0],len(devices)),dtype=np.int32)

        '''
        indexes = np.unique(group, return_index=True)[1]
        no_sort_group = [group[index] for index in sorted(indexes)]
        group = [no_sort_group.index(item) for item in group]
        '''
        new_device_array = device_choice
        ps_or_reduce = np.reshape(ps_or_reduce, (ps_or_reduce.shape[0], 1))
        new_device_array = np.concatenate((ps_or_reduce,new_device_array),axis=1)
        name_list = [nodedef.name for nodedef in self.null_gdef.node]
        strategy = {index_id_dict[index]:new_device_array[self.init_group[index]].tolist() for index in range(len(index_id_dict))}
        strategy = {name: strategy.get(name, list(strategy.values())[0]) for name in name_list}


        _tge = tge.TGE(copy.deepcopy(self.null_gdef), self.devices,sink)

        time_mem_tuple = _tge.custom(strategy).fill_batchsize(self.batch_size).set_nccl_model(self.nccl_model).use_collective().set_bandwidth(self.intra,self.inter).evaluate(self.name_cost_dict)
        time = time_mem_tuple[0]
        mem_list = time_mem_tuple[1]
        time = float(time)/(10**3)

        if any(np.array(mem_list) > np.array(device_mems)):
            time = time*10
            out_of_memory=True
        #reward = np.sum(strategy*strategy)

        if time<self.best_strategy["time"] and out_of_memory==False and record_best:
            self.best_strategy["time"] = time
            self.best_strategy["strategy"] = strategy
            self.best_strategy["group"] = self.init_group
            with open(self.folder_path+"/best_time.log", "w") as f:

                cost_dict=dict()
                for key, value in self.name_cost_dict.items():
                    name = key[0]
                    replica_num=key[1]
                    if replica_num==1:
                        cost_dict[name] = value
                self.best_strategy["cost"] = cost_dict
                json.dump(self.best_strategy.copy(), f)

            best_graph_def = tge.TGE(copy.deepcopy(self.null_gdef), self.devices, self.sink).custom(strategy).replace_placeholder(self.batch_size).use_collective().compile().get_result()
            with open(self.folder_path+"/best_graph.pbtxt", "w") as f:
                f.write(str(best_graph_def))

        if record:
            record_graph_def = tge.TGE(copy.deepcopy(self.null_gdef), self.devices, self.sink).custom(strategy).replace_placeholder(self.batch_size).use_collective().compile().get_result()
            with open(self.folder_path+"/"+record_name, "w") as f:
                f.write(pbtf.MessageToString(record_graph_def))

        return -np.float32(np.sqrt(time)),out_of_memory

    def get_name_cost_dict(self):
        with open(self.folder_path+"/new_cost.pkl", "rb") as f:
            name_cost_dict = pkl.load(f)
        return name_cost_dict



class Graph_item():
    def __init__(self,folder_path,sink):
        self.dataset = load_cora(folder_path,NewWhiteSpaceTokenizer())
        adj = self.dataset.adj_matrix(sparse=True)
        feature_matrix, feature_masks = self.dataset.feature_matrix(bag_of_words=False, sparse=False)

        if "data/graph7" in folder_path:
            self.batch_size = batch_sizes[1]
        elif "data/graph8" in folder_path:
            self.batch_size = batch_sizes[2]
        else:
            self.batch_size = batch_sizes[0]

        self.sink = sink
        if "graph1" in folder_path:
            self.master=True
        else:
            self.master = False

        ####preprocess features################
        feature_matrix = StandardScaler().fit_transform(feature_matrix)
        self.nb_nodes = feature_matrix.shape[0]
        '''
        self.pre_nb_nodes = feature_matrix.shape[0]
        pad_num = 0 if self.pre_nb_nodes%bsz==0 else bsz-self.pre_nb_nodes%bsz
        if pad_num:
            feature_matrix = np.pad(feature_matrix,((0,pad_num),(0,0)),"constant")
        self.nb_nodes = self.pre_nb_nodes+pad_num

        indptr = np.pad(adj.indptr, (0, pad_num), "edge")
        adj = csr_matrix((adj.data, adj.indices, indptr), shape=(self.nb_nodes,self.nb_nodes))
        '''
        self.ft_size = feature_matrix.shape[1]
        self.need_sample = False

        self.biases = process.preprocess_adj_bias(adj)


        self.index_id_dict = self.dataset.network.get_indexer(N_TYPE_NODE).index_id_dict
        self.features = feature_matrix[np.newaxis]


        self.gdef = graph_pb2.GraphDef()
        with open(folder_path+"/null_graph.pbtxt","r")as f:
            txt = f.read()
        pbtf.Parse(txt,self.gdef)

        #####process init group#####
        if os.path.exists(folder_path+"/init_group.json"):
            with open(folder_path+"/init_group.json","r") as f:
                self.init_group =json.load(f)
        else:
           # self.init_group = tge.TGE(copy.deepcopy(self.gdef ), devices, sink).get_groups()
            self.init_group = self.get_colocation_group()
            with open(folder_path+"/new_cost.pkl", "rb") as f:
                name_cost_dict = pkl.load(f)
            self.init_group = group_around_topk_costs(self.gdef,self.init_group,name_cost_dict,group_num)
            with open(folder_path+"/init_group.json","w") as f:
                json.dump(self.init_group,f)
        #print(self.init_group)

        ########################create simulator###########################################
        self.env = Environment(folder_path+"/null_graph.pbtxt",devices,folder_path,self.batch_size,self.init_group,sink)
        self.average_reward=0
        self.best_reward = 1-sys.maxsize
        self.best_replica_num = list()
        self.best_device_choice = np.zeros(shape=(self.nb_nodes, len(devices)), dtype=np.int32)
        self.best_ps_or_reduce = list()
        self.folder_path = folder_path
        ##############create strategy pool#############################
        self.strategy_pool = strategy_pool(folder_path,self.nb_nodes,self.index_id_dict,self.env,self.batch_size,self.init_group,self.sink)
        self.best_group= self.strategy_pool.choose_strategy()["group"] if self.strategy_pool.choose_strategy()!=None else np.arange(max(self.init_group)+1)
        self.avg = None
        self.oom = []
        self.train_place = False
        self.counter=0
        self.small_co = 0.001*5*5
        self.large_co =self.small_co*50
        self.co_entropy = self.small_co
        self.place_lr = lr
        self.record_time =[]
        self.mems = [np.zeros([128, bsz, d_model], dtype=np.float32) for layer in range(n_layer)]
    def get_colocation_group(self):
        leaders = []
        group = []
        for i, nodedef in enumerate(self.gdef.node):
            try:
                colocation_list = nodedef.attr["_class"].list.s
                if len(colocation_list)==0:
                    colocation_name = nodedef.name
                else:
                    colocation  = colocation_list[0]
                    colocation_name = colocation.decode().split("@")[-1]
                if colocation_name  not in leaders:
                    leaders.append(colocation_name)
                    group.append(len(leaders))
                else:
                    group.append(leaders.index(colocation_name))
            except Exception as e:
                traceback.print_exc()
                time.sleep(1)
        print(leaders)
        print(len(leaders))
        print(len(group))
        return group
    def set_session_and_network(self,sess,place_gnn):
        self.sess =sess
        self.place_gnn = place_gnn






    def sample(self,epoch):

        global sample_prob
        sample_prob = min(0.1+0.1*(epoch//60),0.9)

        print("[{}] sample_prob = {}".format(self.folder_path, sample_prob))

        self.replica_masks = mp.Manager().list(range(sample_times+1))
        self.device_choices = mp.Manager().list(range(sample_times+1))
        self.rewards = mp.Manager().list(range(sample_times+1))
        self.ps_or_reduces = mp.Manager().list(range(sample_times+1))
        self.group =mp.Manager().list(range(sample_times+1))
        self.oom = mp.Manager().list(range(sample_times+1))


        self.outputs = self.place_gnn.get_replica_num_prob(
            ftr_in=self.features,
            bias_in=self.biases,
            nb_nodes=self.nb_nodes,
            mems=self.mems,
            init_group = self.init_group)

    def parallel_process_output_unit(self,i):
        def random_func1(output):
            return np.array(list(map(random_choice, output)))

        def random_choice(item):
            np.random.seed()
            choice = []
            choice.append(np.random.choice(item.size, p=item))
            choice.append(np.random.randint(0, item.size))
            return choice[np.random.choice(2, p=[sample_prob, 1 - sample_prob])]

        def sample_func1(output):
            return np.array(list(map(sample_choice, output)))

        def sample_choice(item):
            return np.random.choice(item.size, p=item)

        def argmax_func1(output):
            return np.array(list(map(argmax_choice, output)))

        def argmax_choice(item):
            choice1 = np.argmax(item)
            return choice1
        ti = time.time()
        output = self.outputs[0:len(devices)]
        device_choice = np.zeros(shape=(len(output),output[0].shape[0]))
        if i == sample_times:
            #device_choice = np.array(list(map(argmax_func1, output)))
            for j in range(device_choice.shape[0]):
                for k in range(device_choice.shape[1]):
                    device_choice[j][k] = argmax_choice(output[j][k])
            print(self.folder_path, "argmax_choice0:", time.time() - ti)
        else:
            np.random.seed()
            sample_or_not = True if np.random.choice(2, p=[sample_prob,1-sample_prob])==0 else False
            if sample_or_not:
                #device_choice = np.array(list(map(sample_func1, output)))
                for j in range(device_choice.shape[0]):
                    for k in range(device_choice.shape[1]):
                        device_choice[j][k] = sample_choice(output[j][k])
                print(self.folder_path, "sample_choice0:", time.time() - ti)

            else:
                #device_choice = np.array(list(map(random_func1, output)))
                #for j in range(device_choice.shape[0]):
                #    for k in range(device_choice.shape[1]):
                #        device_choice[j][k] = random_choice(output[j][k])
                device_choice = np.random.randint(0, output[0].shape[1], size=device_choice.shape)
                print(self.folder_path, "random_choice0:", time.time() - ti)

        print(self.folder_path,device_choice.shape)

        ti = time.time()
        device_choice = np.transpose(device_choice)  # from shape[device_num , group_num] to [group_num, device_num]
        device_choice, replica_mask = post_process_device_choice(device_choice, self.batch_size)
        print(self.folder_path,"time1:",time.time()-ti)
        ti = time.time()
        if i == sample_times:
            ps_or_reduce = np.array(list(map(argmax_choice, self.outputs[len(devices)])))
        else:
            if sample_or_not:
                ps_or_reduce = np.array(list(map(sample_choice, self.outputs[len(devices)])))
            else:
                #ps_or_reduce = np.array(list(map(random_choice, self.outputs[len(devices)])))
                ps_or_reduce = np.random.randint(0, 2, size=(self.outputs[len(devices)].shape[0],))
        # ps_or_reduce = self.outputs[max_replica_num]
        # group =  np.array(list(map(random_func1,self.outputs[-1])))
        group = np.array(self.init_group)
        print(self.folder_path,"time2:",time.time()-ti)
        ti = time.time()
        _reward, out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict, self.sink, group)
        print(self.folder_path,"time3:", time.time() - ti)
        if not out_of_memory:
            self.oom[i]=(False)
        else:
            self.oom[i]=(True)

        self.rewards[i]=(_reward)
        self.ps_or_reduces[i]=(ps_or_reduce)
        self.device_choices[i]=(device_choice)
        self.group[i]=(group)
        self.replica_masks[i]=(replica_mask)



    def parallel_process_output(self):
        thres = []
        for i in range(sample_times+1):
            p=mp.Process(target=self.parallel_process_output_unit, args=(i,))
            thres.append(p)
        for p in thres:
            p.start()
        for p in thres:
            p.join()
        print("{} finish!".format(self.folder_path))

        #print("Group:",self.group[0])
    def post_parallel_process(self):
        for i in range(sample_times+1):
            if self.rewards[i] > self.best_reward:
                self.best_reward = self.rewards[i]
                self.best_replica_num = list()
                self.best_device_choice = self.device_choices[i]
                self.best_ps_or_reduce = self.ps_or_reduces[i]
                self.best_group = self.group[i]
            if not self.oom[i]:
                self.strategy_pool.insert(self.rewards[i], self.device_choices[i], self.replica_masks[i], self.ps_or_reduces[i],self.group[i])

    def compute_gradients(self,epoch):
        self.avg = np.mean(self.rewards) if self.avg==None else (self.avg+np.mean(self.rewards))/2
        #print("[{}] train_place = {}".format(self.folder_path, self.train_place))
        #print("[{}] Rewards = {}".format(self.folder_path, self.rewards))
        #print("[{}] epoch = {}".format(self.folder_path, epoch))
        tmp_gradients = []
        for index in range(sample_times):
            #logger.debug("sample_ps:{}".format(self.ps_or_reduces[index]))
            #logger.debug("sample_device_choices:{}".format(self.device_choices[index]))

            place_loss,l2_loss,new_loss,new_global_mems,entropy,gradients=self.place_gnn.get_gradients(ftr_in=self.features,
                            bias_in=self.biases,
                            nb_nodes=self.nb_nodes,
                            sample_ps_or_reduce = np.array(self.ps_or_reduces[index]),
                            sample_device_choice = np.array(self.device_choices[index]),
                            time_ratio = 0.1*self.avg/self.rewards[index],
                            coef_entropy=self.co_entropy,
                            mems = self.mems,
                            init_group=self.init_group,
                            place_lr = self.place_lr)
            self.mems = new_global_mems
            self.cal_entropy = entropy
            #print("place entropy:", self.cal_entropy)
            #print("coefficient:",self.co_entropy)
            #print("l2_loss:",l2_loss)
            #print("entropy loss:",-self.cal_entropy*self.co_entropy)
            #print("place loss:",place_loss)
            #print("place+entropy loss:",new_loss)
            #print("time ratio:",0.1*self.avg/self.rewards[index])
            tmp_gradients.append(gradients)

        times = max(self.rewards)*max(self.rewards)
        self.record_time.append(str(times))
        if len(self.record_time)>20:
            self.record_time = self.record_time[-20:]
        if len(self.record_time)==20 and len(set(self.record_time))<5:
            self.co_entropy = self.large_co
        else:
            self.co_entropy = self.small_co
        if epoch % show_interval == 0:
            print("[{}] step = {}".format(self.folder_path,epoch))
            print("[{}] time = {}".format(self.folder_path,times))
            print("[{}] average reward = {}".format(self.folder_path,self.avg))
            print("[{}] overall entropy:{}".format(self.folder_path,self.cal_entropy))
            with open(self.folder_path+"/time.log", "a+") as f:
                f.write(str(times) + ",")
            with open(self.folder_path+"/entropy.log", "a+") as f:
                f.write(str(self.cal_entropy) + ",")
            with open(self.folder_path+"/loss.log", "a+") as f:
                f.write("place loss:{},entropy loss:{},place+entropy loss:{},l2_loss:{}\n".format(place_loss,-self.cal_entropy*self.co_entropy,new_loss,l2_loss))

        if epoch % show_interval == 0:
            pool_strategy = self.strategy_pool.choose_strategy()
            if pool_strategy==None:
                return self.compute_average_gradients(tmp_gradients)

            place_loss, l2_loss,new_loss,new_global_mems,entropy,gradients=self.place_gnn.get_gradients(ftr_in=self.features,
                            bias_in=self.biases,
                            nb_nodes=self.nb_nodes,
                            sample_ps_or_reduce = np.array(pool_strategy["ps_or_reduce"]),
                            sample_device_choice = np.array(pool_strategy["device_choice"]),
                            time_ratio = 0.1*self.avg/pool_strategy["reward"],
                            coef_entropy=self.co_entropy,
                            mems = self.mems,
                            init_group=self.init_group,
                            place_lr = self.place_lr
                            )
            self.mems = new_global_mems
            self.cal_entropy = entropy
            #print("place entropy:", self.cal_entropy)
            #print("coefficient:",self.co_entropy)
            #print("l2_loss:",l2_loss)
            #print("entropy loss:",-self.cal_entropy*self.co_entropy)
            #print("place loss:",place_loss)
            #print("place+entropy loss:",new_loss)
            #print("time ratio:",0.1*self.avg/pool_strategy["reward"])
            tmp_gradients.append(gradients)

        return self.compute_average_gradients(tmp_gradients)
    def compute_average_gradients(self,tmp_gradients):
        for i,gradient in enumerate(tmp_gradients):
            if i == 0:
                # print type(actor_gradient), len(actor_gradient), type(actor_gradient[0]), len(actor_gradient[0])
                average_gradient = gradient
            else:
                for j in range(0, len(gradient)):
                    average_gradient[j] += gradient[j]
        for j in range(0, len(average_gradient)):
            average_gradient[j] = average_gradient[j] / len(tmp_gradients)
        return average_gradient
    def train(self,epoch):

        self.avg = np.mean(self.rewards) if self.avg==None else (self.avg+np.mean(self.rewards))/2
        print("[{}] train_place = {}".format(self.folder_path, self.train_place))
        print("[{}] Rewards = {}".format(self.folder_path, self.rewards))
        print("[{}] epoch = {}".format(self.folder_path, epoch))

        for index in range(sample_times):
            #logger.debug("sample_ps:{}".format(self.ps_or_reduces[index]))
            #logger.debug("sample_device_choices:{}".format(self.device_choices[index]))

            place_loss,l2_loss,new_loss,new_global_mems,entropy=self.place_gnn.learn_place(ftr_in=self.features,
                            bias_in=self.biases,
                            nb_nodes=self.nb_nodes,
                            sample_ps_or_reduce = np.array(self.ps_or_reduces[index]),
                            sample_device_choice = np.array(self.device_choices[index]),
                            time_ratio = 0.01*self.avg/self.rewards[index],
                            coef_entropy=self.co_entropy,
                            mems = self.mems,
                            init_group=self.init_group,
                            place_lr = self.place_lr)
            self.mems = new_global_mems
            self.cal_entropy = entropy
            print("place entropy:", self.cal_entropy)
            print("coefficient:",self.co_entropy)
            print("l2_loss:",l2_loss)
            print("entropy loss:",-self.cal_entropy*self.co_entropy)
            print("place loss:",place_loss)
            print("place+entropy loss:",new_loss)
            print("time ratio:",0.01*self.avg/self.rewards[index])


        times = max(self.rewards)*max(self.rewards)
        self.record_time.append(str(times))
        if len(self.record_time)>20:
            self.record_time = self.record_time[-20:]
        if len(self.record_time)==20 and len(set(self.record_time))<5:
            self.co_entropy = self.large_co
        else:
            self.co_entropy = self.small_co
        if epoch % show_interval == 0:
            print("[{}] step = {}".format(self.folder_path,epoch))
            print("[{}] time = {}".format(self.folder_path,times))
            print("[{}] average reward = {}".format(self.folder_path,self.avg))
            print("[{}] overall entropy:{}".format(self.folder_path,self.cal_entropy))
            with open(self.folder_path+"/time.log", "a+") as f:
                f.write(str(times) + ",")
            with open(self.folder_path+"/entropy.log", "a+") as f:
                f.write(str(self.cal_entropy) + ",")
            with open(self.folder_path+"/loss.log", "a+") as f:
                f.write("place loss:{},entropy loss:{},place+entropy loss:{},l2_loss:{}\n".format(place_loss,-self.cal_entropy*self.co_entropy,new_loss,l2_loss))

        if epoch % show_interval == 0:
            pool_strategy = self.strategy_pool.choose_strategy()
            if pool_strategy==None:
                return

            place_loss, l2_loss,new_loss,new_global_mems,entropy=self.place_gnn.learn_place(ftr_in=self.features,
                            bias_in=self.biases,
                            nb_nodes=self.nb_nodes,
                            sample_ps_or_reduce = np.array(pool_strategy["ps_or_reduce"]),
                            sample_device_choice = np.array(pool_strategy["device_choice"]),
                            time_ratio = 0.01*self.avg/pool_strategy["reward"],
                            coef_entropy=self.co_entropy,
                            mems = self.mems,
                            init_group=self.init_group,
                            place_lr = self.place_lr
                            )
            self.mems = new_global_mems
            self.cal_entropy = entropy
            print("place entropy:", self.cal_entropy)
            print("coefficient:",self.co_entropy)
            print("l2_loss:",l2_loss)
            print("entropy loss:",-self.cal_entropy*self.co_entropy)
            print("place loss:",place_loss)
            print("place+entropy loss:",new_loss)
            print("time ratio:",0.01*self.avg/pool_strategy["reward"])


class new_place_GNN():
    def __init__(self,sess,ft_size):
        self.first_time = True
        with tf.name_scope('place_gnn'):
            self.sess = sess

            self.ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, None, ft_size),name="ftr_in")
            #self.bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, None, None),name="bias_in")
            self.bias_in = tf.sparse_placeholder(dtype=tf.float32)
            self.nb_node = tf.placeholder(dtype=tf.int32, shape=(),name="nb_node")
            self.attn_drop = tf.placeholder(dtype=tf.float32, shape=(),name="attn_drop")
            self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=(),name="ffd_drop")
            self.is_train = tf.placeholder(dtype=tf.bool, shape=(),name="is_train")
            self.init_group = tf.placeholder(dtype=tf.int32, shape=(None,),name="init_group")
            self.sample_ps_or_reduce = tf.placeholder(dtype=tf.int32, shape=(None,),name="sample_ps_or_reduce")
            self.sample_device_choice = tf.placeholder(dtype=tf.int32, shape=(None,len(devices),),name="sample_device_choice")
            self.previous_device_choices = tf.placeholder(dtype=tf.float32, shape=(len(devices),None,max_replica_num+1),name="previous_device_choices")
            self.time_ratio = tf.placeholder(dtype=tf.float32, shape=(),name="time_ratio")
            self.coef_entropy = tf.placeholder(dtype=tf.float32, shape=(),name="coef_entropy")

            self.train_place = tf.placeholder(dtype=tf.bool, shape=(),name="train_place")
            self.mems = [tf.placeholder(tf.float32,[128, bsz, d_model]) for _ in range(n_layer)]
            self.place_lr = tf.placeholder(dtype=tf.float32, shape=(),name="place_lr")


        with tf.variable_scope("group_nn"):
            group = self.init_group
            self.group = group
            unique_group,_ =tf.unique(group)
            self.unique_group = unique_group

        with tf.variable_scope("place_nn"):
            with tf.device("/device:GPU:1"):
                logits = model.inference(self.ftr_in, 1024, self.nb_node, self.is_train,
                                         self.attn_drop, self.ffd_drop,
                                         bias_mat=self.bias_in,
                                         hid_units=hid_units, n_heads=n_heads,
                                         residual=residual, activation=nonlinearity)

            with tf.device("/device:GPU:0"):
                logits = model.inference(logits, d_model, self.nb_node, self.is_train,
                                         self.attn_drop, self.ffd_drop,
                                         bias_mat=self.bias_in,
                                         hid_units=place_hid_units, n_heads=place_n_heads,
                                         residual=residual, activation=nonlinearity)
            logits = tf.reshape(logits, [-1, d_model])
            self.logits_before = logits
            logits = tf.unsorted_segment_max(logits, self.init_group, tf.reduce_max(self.init_group) + 1)
            logits = tf.reshape(logits, [-1, d_model])

            #logits=tf.unsorted_segment_sum(logits, group,tf.reduce_max(group)+1)
            self.logits =logits
            self.device_choices = list()
            self.log_device_choices = list()
            log_resh = tf.reshape(logits, [-1,bsz,d_model])
            initializer = tf.initializers.random_normal(
                stddev=0.02,
                seed=None)
            output,self.new_mems = transformer(log_resh,self.mems,n_layer,d_model,n_head,d_head,d_inner,0.1,0.1,initializer,True,mem_len=128)
            output = tf.reshape(output, [-1,d_model])
            #output = output[:,:(max_replica_num+1)*(len(devices))+2]

            output = tf.layers.dense(output,(max_replica_num+1)*(len(devices))+2)

            sum = 0
            for i in range(0,len(devices)):
                oi = tf.nn.softmax(output[:,i*(max_replica_num+1):(i+1)*(max_replica_num+1)])
                self.device_choices.append(oi)
                #log_oi = tf.nn.log_softmax(output[:,i*(max_replica_num+1):(i+1)*(max_replica_num+1)])
                log_oi = tf.log(oi+10e-8)
                self.log_device_choices.append(log_oi)
                sum = sum + tf.reduce_sum((log_oi* oi))
            ps_or_reduce_prob = tf.nn.softmax(output[:,-2:])
            self.ps_or_reduce = ps_or_reduce_prob
            #self.log_ps_reduce = tf.nn.log_softmax(output[:,-2:])
            self.log_ps_reduce = tf.log( self.ps_or_reduce+10e-8)
            self.entropy = tf.reduce_sum((self.log_ps_reduce * ps_or_reduce_prob), 1)
            self.entropy = -(tf.reduce_sum(self.entropy) + sum )

            _range = tf.range(tf.shape(self.sample_ps_or_reduce)[0])[:, tf.newaxis]
            kl = 0
            KLDivergence = tf.keras.losses.KLDivergence()
            for k in range(0,len(devices)):
                kl+=KLDivergence(self.device_choices[i],self.previous_device_choices[i])
            self.place_kl = kl

            self.place_reward = []

            indices = tf.concat((_range, self.sample_ps_or_reduce[:, tf.newaxis]), axis=1)
            log_prob = tf.gather_nd(self.log_ps_reduce, indices)
            #log_prob = tf.gather(log_prob,unique_group)
            self.place_reward.append(tf.reduce_sum(log_prob )* self.time_ratio)


            #rest device choice n*(m+1)
            for j in range(0,len(devices)):
                indices = tf.concat((_range, self.sample_device_choice[:, j][:, tf.newaxis]), axis=1)
                log_prob = tf.gather_nd(self.log_device_choices[j], indices)
                self.before_log_prob =self.log_device_choices[j]
                self.indices = indices
                self.log_prob = log_prob
                self.place_reward.append(tf.reduce_sum(log_prob) * self.time_ratio)
            self.place_reward = tf.add_n(self.place_reward)

        place_reward =  self.place_reward+self.coef_entropy * self.entropy
        #self.loss = -reward
        self.place_loss = -place_reward
        self.network_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='place_nn')

        self.loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.network_params if v.name not in ['bias', 'gamma', 'b', 'g', 'beta']]) * l2_coef

        self.net_gradients = tf.compat.v1.gradients(self.place_loss+self.loss_l2, self.network_params,colocate_gradients_with_ops=True)
        self.apply_grad = tf.train.AdamOptimizer(learning_rate=self.place_lr,beta1=0.9,beta2=0.98, epsilon=1e-9).apply_gradients(zip(self.net_gradients,self.network_params))

        #self.train_place_op,self.loss_l2 = model.training(self.place_loss,self.place_lr , l2_coef,vars=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='place_nn'))


    def get_gradients(self,ftr_in,bias_in,nb_nodes,sample_ps_or_reduce,sample_device_choice,time_ratio,coef_entropy,mems,init_group,place_lr):
        feed_dict = {}
        feed_dict[self.ftr_in]=ftr_in
        feed_dict[self.bias_in]=bias_in
        feed_dict[self.nb_node]=nb_nodes
        feed_dict[self.is_train]=True
        feed_dict[self.attn_drop]=0.1
        feed_dict[self.ffd_drop]=0.1
        feed_dict[self.sample_ps_or_reduce]=sample_ps_or_reduce
        feed_dict[self.sample_device_choice]=sample_device_choice
        feed_dict[self.time_ratio]=time_ratio
        feed_dict[self.coef_entropy]=coef_entropy
        feed_dict[self.previous_device_choices]=self.previous_outputs[:len(devices)]
        feed_dict[self.train_place] = True


        feed_dict[self.place_lr]=place_lr

        feed_dict[self.init_group]=init_group

        for item1,item2 in zip(self.mems,mems):
            feed_dict[item1]=item2

        fetch_list =[item for item in self.device_choices]
        fetch_list.extend([self.place_reward,self.loss_l2,self.before_log_prob,self.entropy,self.place_loss,self.new_mems,self.net_gradients])

        outputs= self.sess.run(fetch_list,
                     feed_dict=feed_dict)

        place_reward,l2_loss,before_log_prob, entropy, loss, mems, gradients = outputs[len(devices):]
        #logger.debug("previous_outputs:{}".format(self.previous_outputs[:len(devices)]))

        return -place_reward,l2_loss,loss,mems,entropy,gradients

    def apply_gradients(self,gradients,place_lr):
        feed_dict = {i:d for i,d in zip(self.net_gradients,gradients)}
        feed_dict[self.place_lr] = place_lr
        _= self.sess.run(self.apply_grad,
                     feed_dict=feed_dict)

    def learn_place(self,ftr_in,bias_in,nb_nodes,sample_ps_or_reduce,sample_device_choice,time_ratio,coef_entropy,mems,init_group,place_lr):
        feed_dict = {}
        feed_dict[self.ftr_in]=ftr_in
        feed_dict[self.bias_in]=bias_in
        feed_dict[self.nb_node]=nb_nodes
        feed_dict[self.is_train]=True
        feed_dict[self.attn_drop]=0.1
        feed_dict[self.ffd_drop]=0.1
        feed_dict[self.sample_ps_or_reduce]=sample_ps_or_reduce
        feed_dict[self.sample_device_choice]=sample_device_choice
        feed_dict[self.time_ratio]=time_ratio
        feed_dict[self.coef_entropy]=coef_entropy
        feed_dict[self.previous_device_choices]=self.previous_outputs[:len(devices)]
        feed_dict[self.train_place] = True


        feed_dict[self.place_lr]=place_lr

        feed_dict[self.init_group]=init_group

        for item1,item2 in zip(self.mems,mems):
            feed_dict[item1]=item2


        fetch_list =[item for item in self.device_choices]
        fetch_list.extend([self.place_reward,self.loss_l2,self.before_log_prob,self.entropy,self.unique_group,self.log_device_choices[0],self.log_prob,self.indices,self.place_loss,self.new_mems,self.train_place_op])

        outputs= self.sess.run(fetch_list,
                     feed_dict=feed_dict)
        self.previous_outputs = outputs

        place_reward,l2_loss,before_log_prob, entropy, unique, log_device_choices, log_prob, indices, loss, mems, _ = outputs[len(devices):]
        #logger.debug("previous_outputs:{}".format(self.previous_outputs[:len(devices)]))

        return -place_reward,l2_loss,loss,mems,entropy


    def get_replica_num_prob(self,ftr_in,bias_in,nb_nodes,mems,init_group):
        fetch_list =[item for item in self.device_choices]
        fetch_list.append(self.ps_or_reduce)
        fetch_list.append(self.logits_before)
        fetch_list.append(self.logits)
        fetch_list.append(self.group)



        feed_dict = {}
        feed_dict[self.ftr_in]=ftr_in
        feed_dict[self.bias_in]=bias_in
        feed_dict[self.nb_node]=nb_nodes
        feed_dict[self.is_train]=False
        feed_dict[self.attn_drop]=0
        feed_dict[self.ffd_drop]=0
        feed_dict[self.init_group] = init_group
        for item1,item2 in zip(self.mems,mems):
            feed_dict[item1]=item2

        outputs = self.sess.run(fetch_list, feed_dict=feed_dict)
        print("device choice prob:", outputs[0])
        #print("Logits after:",outputs[-4])
        #print("Group:",outputs[-3])
        if self.first_time:
            self.previous_outputs = outputs[:len(devices)]
            self.first_time=False
        return outputs


def main_entry():
    models = []
    for i,feature_folder in enumerate(feature_folders):
        item = Graph_item(feature_folder,sinks[i])
        models.append(item)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        place_gnn = new_place_GNN(sess,ft_size=models[0].ft_size)

        saver = tf.train.Saver()
        try:
            saver.restore(sess, checkpt_file)
        except:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

        for model in models:
            model.set_session_and_network(sess,place_gnn)


        for epoch in range(nb_epochs):
            for model in models:
                model.sample(epoch)


            processes=[]
            for model in models:
                processes.append(mp.Process(target=model.parallel_process_output))
                #model.parallel_process_output()
            for pro in processes:
                pro.start()
            for pro in processes:
                pro.join()

            gradients = []
            for model in models:
                model.post_parallel_process()
                #model.train(epoch)
                gradients.append(model.compute_gradients(epoch))

            for i, gradient in enumerate(gradients):
                #print(type(gradient), len(gradient))
                #print(type(gradient[0]), len(gradient[0]))

                if i == 0:
                    average_gradient = gradient
                else:
                    for j in range(0, len(gradient)):
                        average_gradient[j] += gradient[j]
            for j in range(0, len(average_gradient)):
                average_gradient[j] = average_gradient[j] / len(gradients)
            #print("Gradients:",average_gradient)
            place_gnn.apply_gradients(average_gradient,lr)

            if epoch % (show_interval*30 )== 0:
                saver.save(sess, checkpt_file)

        sess.close()


if __name__ == '__main__':

    main_entry()