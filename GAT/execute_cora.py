import time
import numpy as np
import tensorflow as tf

from models import GAT
from models import SpGAT
from gat_utils import process
from data_process.dataset import GraphDataset, WhiteSpaceTokenizer,NewWhiteSpaceTokenizer
from data_process.example import load_M10, load_cora, load_dblp
from data_process.meta_network import MetaNetwork, N_TYPE_NODE, N_TYPE_LABEL, IdIndexer
import google.protobuf.text_format as pbtf
from tensorflow.core.framework import graph_pb2
from sklearn.preprocessing import StandardScaler
import copy
import sys
import os
import scipy.sparse as sp

import pickle
sys.path.append('../')
import tge
import json
import pickle as pkl
import multiprocessing as mp
from multiprocessing import Pool


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
os.environ["TF_XLA_FLAGS"]="--tf_xla_cpu_global_jit"
batch_size = 1
nb_epochs = 100000
patience = 100
lr = config_dict.get("learning_rate", 0.01)  # learning rate
l2_coef = 0  # weight decay
hid_units = [1024]  # numbers of hidden units per each attention head in each layer
n_heads = [4, 4]  # additional entry for the output layer
place_hid_units = [256, 256,512,512,256, 256]
place_n_heads = [4,4,4,2,2,2, 1]
residual = False
nonlinearity = tf.nn.elu
model = SpGAT
transformer = True
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
feature_folders = config_dict.get("inputs",["data/graph1", "data/graph2", "data/graph3", "data/graph4", "data/graph5", "data/graph6","data/graph7"])
sinks =  config_dict.get("sinks",[["GradientDescent"], ["GradientDescent"], ["GradientDescent"], ["GradientDescent"], ["GradientDescent"], ["GradientDescent"],["group_deps_1","loss/Mean","global_step/add"]])
sample_times = 1
devices = config_dict.get("devices", [
    "/job:tge/replica:0/task:0/device:GPU:0",
    "/job:tge/replica:0/task:0/device:GPU:1",
    "/job:tge/replica:0/task:1/device:GPU:0",
    "/job:tge/replica:0/task:1/device:GPU:1"
])

max_replica_num = config_dict.get("max_replica_num", len(devices))
show_interval = 3
num_cores = mp.cpu_count()
device_mems = config_dict.get("device_mems", [16 * 10e9, 16 * 10e9, 16 * 10e9, 16 * 10e9])
def find_index(array, item):
    for idx, val in enumerate(array):
        if val == item:
            return idx
    return len(array)
def post_func1(item):
    item1=item[:len(item)-1]
    batch_size = item[-1]
    replica_num = find_index(item1, len(devices))
    while(batch_size%replica_num):
        replica_num-=1
    if replica_num<len(item1):
        item1[replica_num]=len(devices)
        if replica_num < len(item1)-1:
            item1[replica_num+1:] = -1
    return item1
def post_func2(item1):
    replica_mask = np.ones(shape=len(item1),dtype=np.int32)
    replica_num = find_index(item1, len(devices))
    if replica_num < len(item1)-1:
        replica_mask[replica_num + 1:] = 0
    return replica_mask
def post_process_device_choice(device_choice,batch_size):

    #post process and align to batch size
    new_batch_size=np.ones(shape=(device_choice.shape[0],1)) * batch_size
    device_choice=np.array(list(map(post_func1,np.concatenate((device_choice,new_batch_size),axis=1))),dtype=np.int32)
    replica_mask = np.array(list(map(post_func2,device_choice)),dtype=np.int32)
    return device_choice,replica_mask


def rel_multihead_attn(w, r,d_model,mems,
                       n_head, d_head, nb_nodes,scope='rel_attn'):
    scale = 1 / (d_head ** 0.5)
    cat = w
    if False:
        w = tf.expand_dims(w, axis=0)
        seq_fts = tf.layers.conv1d(w, d_model, 1, use_bias=False)

        # simplest self-attention possible
        f_1 = tf.layers.conv1d(seq_fts, 1, 1)
        f_2 = tf.layers.conv1d(seq_fts, 1, 1)

        f_1 = tf.reshape(f_1, (nb_nodes, 1))
        f_2 = tf.reshape(f_2, (nb_nodes, 1))

        f_1 = r * f_1

        f_2 = r * tf.transpose(f_2, [1, 0])


        logits = tf.sparse_add(f_1, f_2)
        lrelu = tf.SparseTensor(indices=logits.indices,
                                values=tf.nn.leaky_relu(logits.values),
                                dense_shape=logits.dense_shape)
        coefs = tf.sparse_softmax(lrelu)

        # As tf.sparse_tensor_dense_matmul expects its arguments to have rank-2,
        # here we make an assumption that our input is of batch size 1, and reshape appropriately.
        # The method will fail in all other cases!
        coefs = tf.sparse_reshape(coefs, [nb_nodes, nb_nodes])
        seq_fts = tf.reshape(seq_fts,(nb_nodes,seq_fts.shape[-1]))
        vals = tf.sparse_tensor_dense_matmul(coefs, seq_fts)
        vals = tf.reshape(vals,(nb_nodes,seq_fts.shape[-1]))
        w_heads = tf.layers.dense(cat, 3 * n_head * d_head, use_bias=False)
        r_head_k = tf.layers.dense(vals, n_head * d_head, use_bias=False)



        w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)


        rw_head_q = w_head_q

       # AC = tf.einsum('ib,jb->ij', rw_head_q, w_head_k)
        print(rw_head_q.shape)
        print(tf.transpose(w_head_k).shape)

        AC =tf.matmul(rw_head_q,tf.transpose(w_head_k))
        BD =tf.matmul(rw_head_q,tf.transpose(r_head_k))
        print(AC.shape)
        print(BD.shape)
        attn_score = (AC+BD) * scale
        attn_prob = tf.nn.softmax(attn_score, 1)

        attn_vec = tf.matmul(attn_prob, w_head_v)
        # size_t = tf.shape(attn_vec)
        # attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])

        attn_out = tf.layers.dense(attn_vec, d_model, use_bias=False, activation=tf.nn.leaky_relu)
        output = tf.contrib.layers.layer_norm(attn_out + cat, begin_norm_axis=-1)
    else:
        w = tf.expand_dims(w, axis=0)
        qlen = tf.shape(w)[0]
        bsz = tf.shape(w)[1]

        cat = tf.concat([mems, w],
                        0) if mems is not None and mems.shape.ndims > 1 else w
        cat=w
        w_heads = tf.layers.dense(cat, 3 * n_head * d_head, use_bias=False)

        w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
        w_head_q = w_head_q[-qlen:]

        klen = tf.shape(w_head_k)[0]

        w_head_q = tf.reshape(w_head_q, [qlen, bsz, n_head, d_head])
        w_head_k = tf.reshape(w_head_k, [klen, bsz, n_head, d_head])
        w_head_v = tf.reshape(w_head_v, [klen, bsz, n_head, d_head])

        rw_head_q = w_head_q

        AC = tf.einsum('ibnd,jbnd->ijbn', rw_head_q, w_head_k)


        attn_score = (AC) * scale

       # w_heads = tf.layers.dense(cat, 3 * n_head * d_head, use_bias=False)
       # w_head_q, w_head_k, w_head_v = tf.split(w_heads, 3, -1)
       # rw_head_q = w_head_q
       # AC =tf.matmul(rw_head_q,tf.transpose(w_head_k))
       # attn_score = (AC) * scale



        attn_prob = tf.nn.softmax(attn_score, 1)

        #attn_vec = tf.matmul(attn_prob,w_head_v)
        attn_vec = tf.einsum('ijbn,jbnd->ibnd', attn_prob, w_head_v)
        size_t = tf.shape(attn_vec)
        attn_vec = tf.reshape(attn_vec, [size_t[0], size_t[1], n_head * d_head])

        attn_out = tf.layers.dense(attn_vec, d_model, use_bias=False)
        output = tf.contrib.layers.layer_norm(attn_out + cat, begin_norm_axis=-1)
    return output[0]
def comp_fc(item):
    item1 = item[:int(len(item)/2)]
    item2 = item[int(len(item)/2):]
    return 0 if all(item1 == item2) else 1
class strategy_pool(object):
    def __init__(self,folder_path,node_num,index_id_dict,env,batch_size,sink):
        self.folder_path = folder_path
        self.node_num = node_num
        self.index_id_dict = index_id_dict
        self.env = env
        self.sink = sink
        if os.path.exists(self.folder_path+"/pool.pkl"):
            with open(self.folder_path+"/pool.pkl","rb") as f:
                self.strategies= pkl.load(f)
            self.save_strategy_pool()
        else:
            self.strategies = list()

        self.rewards = [item["reward"] for item in self.strategies] if len(self.strategies) else [-sys.maxsize]
        self.batch_size = batch_size

        # even data parallel 1
        #device_choice = np.zeros(shape=(self.node_num, len(devices)), dtype=np.int32)
        device_choice = np.array([np.arange(max_replica_num)%(len(devices)) for i in range(self.node_num)],dtype=np.int32)
        device_choice,replica_mask = post_process_device_choice(device_choice,self.batch_size)
        ps_or_reduce = np.ones(shape=(self.node_num, ), dtype=np.int32)
        reward,out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict,self.sink)
        if not out_of_memory:
            self.insert(reward, device_choice, replica_mask,ps_or_reduce)

        # even data parallel 2
        #device_choice = np.zeros(shape=(self.node_num, len(devices)), dtype=np.int32)
        device_choice = np.negative(np.ones(shape=(self.node_num, max_replica_num), dtype=np.int32))
        for item in device_choice:
            for i in range(len(devices)):
                item[i] =i
            item[len(devices)]=len(devices)

        device_choice,replica_mask = post_process_device_choice(device_choice,self.batch_size)
        ps_or_reduce = np.ones(shape=(self.node_num, ), dtype=np.int32)
        reward,out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict,self.sink)
        if not out_of_memory:
            self.insert(reward, device_choice, replica_mask,ps_or_reduce)


        # even data parallel 3
        #device_choice = np.zeros(shape=(self.node_num, len(devices)), dtype=np.int32)
        device_choice = np.negative(np.ones(shape=(self.node_num, max_replica_num), dtype=np.int32))
        for item in device_choice:
            for i in range(len(devices)):
                item[i] =i
            item[len(devices)]=len(devices)

        device_choice,replica_mask = post_process_device_choice(device_choice,self.batch_size)
        ps_or_reduce = np.zeros(shape=(self.node_num, ), dtype=np.int32)
        reward,out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict,self.sink,record=True,record_name="dp_graph.pbtxt")
        if not out_of_memory:
            self.insert(reward, device_choice, replica_mask,ps_or_reduce)

        #single gpu
        device_choice = np.negative(np.ones(shape=(self.node_num, max_replica_num), dtype=np.int32))
        for item in device_choice:
            item[0] =0
            item[1] = len(devices)

        device_choice,replica_mask = post_process_device_choice(device_choice,self.batch_size)
        ps_or_reduce = np.ones(shape=(self.node_num, ), dtype=np.int32)
        reward,out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict,self.sink,record=True,record_name="single_graph.pbtxt")
        if not out_of_memory:
            self.insert(reward, device_choice, replica_mask,ps_or_reduce)

        #model parallel 1
        device_choice = np.negative(np.ones(shape=(self.node_num, max_replica_num), dtype=np.int32))
        for i,item in enumerate(device_choice):
            item[0] = i%(len(devices))
            item[1] = len(devices)

        device_choice,replica_mask = post_process_device_choice(device_choice,self.batch_size)
        ps_or_reduce = np.ones(shape=(self.node_num, ), dtype=np.int32)
        reward,out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict,self.sink)
        if not out_of_memory:
            self.insert(reward, device_choice, replica_mask,ps_or_reduce)

        # model parallel 2
        device_choice = np.negative(np.ones(shape=(self.node_num, max_replica_num), dtype=np.int32))
        for i, item in enumerate(device_choice):
            item[0] = int(i//(len(device_choice)/(len(devices))))
            item[1] = len(devices)

        device_choice,replica_mask = post_process_device_choice(device_choice,self.batch_size)
        ps_or_reduce = np.ones(shape=(self.node_num,), dtype=np.int32)
        reward, out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict,self.sink)
        if not out_of_memory:
            self.insert(reward, device_choice, replica_mask,ps_or_reduce)

        self.rewards = [item["reward"] for item in self.strategies]
    def get_length(self):
        return len(self.strategies)

    def get_stratey_list(self,device_choice,ps_or_reduce):
        new_device_array = np.array(list(map(reward_func,device_choice)),dtype=np.int32)
        ps_or_reduce = np.reshape(ps_or_reduce, (ps_or_reduce.shape[0], 1))
        new_device_array = np.concatenate((ps_or_reduce,new_device_array),axis=1)
        return new_device_array.tolist()

    def save_strategy_pool(self):
        with open(self.folder_path + "/pool.pkl", "wb") as f:
            pkl.dump(self.strategies,f)
        with open(self.folder_path + "/pool.log", "w") as f:
            f.write(str([item["reward"] for item in self.strategies]))
            for i in range(4):
                f.write("\nstrategy"+str(i)+":\n")
                f.write(str(self.strategies[np.random.randint(len(self.strategies))]["strategy_list"]))

    def get_replica_masks(self,device_choice):
        masks = np.zeros(shape=device_choice.shape,dtype=np.int32)
        for i in range(masks.shape[0]):
            for j in range(masks.shape[1]):
                masks[i,j] = 0 if device_choice[i,j]==-1 else 1
        return masks

    def insert(self,reward,device_choice,replica_mask,ps_or_reduce):
        strategy_list = self.get_stratey_list(device_choice, ps_or_reduce)

        if len(self.strategies)<4:
            for j,strategy in enumerate(self.strategies):
                exist_device_choice = (strategy["device_choice"])
                #diff_list = list(map(comp_fc,np.concatenate((device_choice,exist_device_choice),axis=1)))
                if False:#sum(diff_list)/len(diff_list)<0.05:
                    if reward>strategy["reward"]:
                        self.strategies.append({"replica_mask":replica_mask,"strategy_list":strategy_list,"reward":reward,"device_choice":device_choice,"ps_or_reduce":ps_or_reduce})
                        self.strategies.pop(j)
                        self.save_strategy_pool()
                        self.rewards = [item["reward"] for item in self.strategies]
                    return
            self.strategies.append({"replica_mask": replica_mask, "strategy_list": strategy_list, "reward": reward,
                                    "device_choice": device_choice, "ps_or_reduce": ps_or_reduce})

            self.save_strategy_pool()
            self.rewards.append(reward)

        elif len(self.strategies)<5 and reward>np.mean(self.rewards):
            for j,strategy in enumerate(self.strategies):
                exist_device_choice = (strategy["device_choice"])
                #diff_list = list(map(comp_fc,np.concatenate((device_choice,exist_device_choice),axis=1)))
                if False:#sum(diff_list)/len(diff_list)<0.05:
                    if reward>strategy["reward"]:
                        self.strategies.append({"replica_mask":replica_mask,"strategy_list":strategy_list,"reward":reward,"device_choice":device_choice,"ps_or_reduce":ps_or_reduce})
                        self.strategies.pop(j)
                        self.save_strategy_pool()
                        self.rewards = [item["reward"] for item in self.strategies]
                    return
            self.strategies.append({"replica_mask": replica_mask, "strategy_list": strategy_list, "reward": reward,
                                    "device_choice": device_choice, "ps_or_reduce": ps_or_reduce})

            self.save_strategy_pool()
            self.rewards.append(reward)
        elif len(self.strategies)>=5 and reward>np.mean(self.rewards):
            for j,strategy in enumerate(self.strategies):
                exist_device_choice = (strategy["device_choice"])
                #diff_list = list(map(comp_fc,np.concatenate((device_choice,exist_device_choice),axis=1)))
                if False:#sum(diff_list)/len(diff_list)<0.05:
                    if reward>strategy["reward"]:
                        self.strategies.append({"replica_mask":replica_mask,"strategy_list":strategy_list,"reward":reward,"device_choice":device_choice,"ps_or_reduce":ps_or_reduce})
                        self.strategies.pop(j)
                        self.save_strategy_pool()
                        self.rewards = [item["reward"] for item in self.strategies]
                    return
            index = self.rewards.index(min(self.rewards))
            self.strategies.pop(index)
            self.strategies.append({"replica_mask": replica_mask, "strategy_list": strategy_list, "reward": reward,
                                    "device_choice": device_choice, "ps_or_reduce": ps_or_reduce})

            self.save_strategy_pool()
            self.rewards = [item["reward"] for item in self.strategies]

    def choose_strategy(self):
        self.rewards = [item["reward"] for item in self.strategies]
        index = np.random.randint(0,len(self.strategies))
        index = self.rewards.index(max(self.rewards))
        return self.strategies[index]
def reward_func(item):
    new_device_array = np.zeros(shape=(len(devices)),dtype=np.int32)
    for j in range(len(item)):
        if item[j]!=-1 and item[j]!=len(devices):
            new_device_array[item[j]]+=1
    return new_device_array
class Environment(object):
    def __init__(self,gdef_path,devices,folder_path,batch_size,pool,sink):

        self.gdef = graph_pb2.GraphDef()
        with open(gdef_path,"r")as f:
            txt = f.read()
        pbtf.Parse(txt,self.gdef)
        self.folder_path = folder_path
        self.random_strategy=list()
        self.best_strategy = mp.Manager().dict()
        self.best_strategy["time"] = sys.maxsize
        self.batch_size = batch_size
        self.pool = pool
        self.devices =devices
        self.sink =sink
        if "graph7" in folder_path:
            self.null_gdef =self.gdef
        else:
            self.null_gdef = graph_pb2.GraphDef()
            with open(folder_path+"/null_graph.pbtxt","r")as f:
                txt = f.read()
            pbtf.Parse(txt,self.null_gdef)

        if os.path.exists(folder_path+"/best_time.log"):
            with open(folder_path+"/best_time.log", "r") as f:
                tmp = json.load(f)
                for key,value in tmp.items():
                    self.best_strategy[key] = value
            best_graph_def =tge.TGE(copy.deepcopy(self.null_gdef), self.devices, self.sink).custom(self.best_strategy["strategy"]).compile().get_result()
            with open(self.folder_path+"/best_graph.pbtxt", "w") as f:
                f.write(str(best_graph_def))



        self.name_cost_dict = self.get_name_cost_dict()

    def get_reward2(self,device_choice,ps_or_reduce,index_id_dict,sink,record=False,record_name=None):
        out_of_memory=False
        #new_device_array = np.zeros(shape=(device_choice.shape[0],len(devices)),dtype=np.int32)

        '''
        for i in range(device_choice.shape[0]):
            for j in range(device_choice.shape[1]):
                if device_choice[i,j]!=-1 and device_choice[i,j]!=len(self.devices):
                    new_device_array[i,device_choice[i,j]]+=1
        '''
        new_device_array = np.array(list(map(reward_func,device_choice)))
        ps_or_reduce = np.reshape(ps_or_reduce, (ps_or_reduce.shape[0], 1))
        new_device_array = np.concatenate((ps_or_reduce,new_device_array),axis=1)
        strategy = {index_id_dict[index]:new_device_array[index].tolist() for index in range(new_device_array.shape[0])}
        bandwidth = config_dict.get("bandwidth",None)
        if bandwidth==None:
            intra = "5000"
            inter = "1250"
        else:
            intra = bandwidth[0]
            inter = bandwidth[1]
        _tge = tge.TGE(copy.deepcopy(self.gdef), self.devices,sink)
        time_mem_tuple = _tge.custom(strategy).set_bandwidth(intra,inter).evaluate(self.name_cost_dict)
        time = time_mem_tuple[0]
        mem_list = time_mem_tuple[1]
        time = float(time)/(10**3)

        if any(np.array(mem_list) > np.array(device_mems)):
            time = time*10
            out_of_memory=True
        #reward = np.sum(strategy*strategy)

        if time<self.best_strategy["time"] and out_of_memory==False:
            self.best_strategy["time"] = time
            self.best_strategy["strategy"] = strategy
            with open(self.folder_path+"/best_time.log", "w") as f:

                cost_dict=dict()
                for key, value in self.name_cost_dict.items():
                    name = key[0]
                    batch_size=key[1]
                    if batch_size==self.batch_size:
                        cost_dict[name] = value
                self.best_strategy["cost"] = cost_dict
                json.dump(self.best_strategy.copy(), f)

            best_graph_def = tge.TGE(copy.deepcopy(self.null_gdef), self.devices, self.sink).custom(strategy).compile().get_result()
            with open(self.folder_path+"/best_graph.pbtxt", "w") as f:
                f.write(str(best_graph_def))

        if record:
            record_graph_def = tge.TGE(copy.deepcopy(self.null_gdef), self.devices, self.sink).custom(strategy).compile().get_result()
            with open(self.folder_path+"/"+record_name, "w") as f:
                f.write(str(record_graph_def))

        return -np.float32(np.sqrt(time)),out_of_memory

    def get_name_cost_dict(self):
        with open(self.folder_path+"/cost.pkl", "rb") as f:
            name_cost_dict = pkl.load(f)
        return name_cost_dict


def sample_func1(output):
    return np.array(list(map(lambda x: np.random.choice(x.size, p=x), output)))
def random_func1(item):
    return  np.random.choice(item.size, p=item)
import threading

class feature_item(threading.Thread):
    def __init__(self,folder_path,pool,event,event2,sink):
        super(feature_item, self).__init__()
        self.dataset = load_cora(folder_path,NewWhiteSpaceTokenizer())
        adj = self.dataset.adj_matrix(sparse=True)
        feature_matrix, feature_masks = self.dataset.feature_matrix(bag_of_words=False, sparse=False)
        self.batch_size = int(feature_matrix[0,-1])
        self.event = event
        self.event.set()
        self.event2 = event2
        self.event2.clear()
        self.sink = sink
        feature_matrix = StandardScaler().fit_transform(feature_matrix)

        labels, label_masks = self.dataset.label_list_or_matrix(one_hot=False)

        train_node_indices, test_node_indices, train_masks, test_masks = self.dataset.split_train_and_test(training_rate=0.8)

        self.nb_nodes = feature_matrix.shape[0]
        self.ft_size = feature_matrix.shape[1]
        self.need_sample = False

        self.biases = process.preprocess_adj_bias(adj)

        train_mask=train_masks
        val_mask=test_masks
        test_mask=test_masks

        self.index_id_dict = self.dataset.network.get_indexer(N_TYPE_NODE).index_id_dict
        self.features = feature_matrix[np.newaxis]

        self.train_mask = train_mask[np.newaxis]
        self.val_mask = val_mask[np.newaxis]
        self.test_mask = test_mask[np.newaxis]

        self.pool = pool
        self.env = Environment(folder_path+"/graph.pbtxt",devices,folder_path,self.batch_size,self.pool,sink)
        self.average_reward=0
        self.best_reward = 1-sys.maxsize
        self.best_replica_num = list()
        self.best_device_choice = np.zeros(shape=(self.nb_nodes, max_replica_num), dtype=np.int32)
        self.best_ps_or_reduce = list()
        self.folder_path = folder_path
        self.strategy_pool = strategy_pool(folder_path,self.nb_nodes,self.index_id_dict,self.env,self.batch_size,self.sink)

        self.mems=[np.zeros([2, self.nb_nodes, 64], dtype=np.float32) for layer in range(3)]
        self.mutex = threading.Lock()
        self.avg = float(np.mean(self.strategy_pool.rewards))
        self.oom = []

    def set_session_and_network(self,sess,place_gnn):
        self.sess =sess
        self.place_gnn = place_gnn

    def sample_one_time(self):
        self.sample()
        self.proc = mp.Process(target=self.parallel_process_output, args=())
        self.proc.start()
    def wait_sample(self):
        self.proc.join()

    def sync_sample_and_parallel_process(self):
        self.sample()
        self.parallel_process_output()

    def sample(self):

        self.replica_masks = mp.Manager().list()
        self.device_choices = mp.Manager().list()
        self.rewards = mp.Manager().list()
        self.ps_or_reduces = mp.Manager().list()
        self.oom = mp.Manager().list()
        co_entropy = 0
        self.thres = []
        tr_step = 0
        tr_size = self.features.shape[0]

        self.outputs = self.place_gnn.get_replica_num_prob_and_entropy(
            ftr_in=self.features[tr_step * batch_size:(tr_step + 1) * batch_size],
            bias_in=self.biases,
            nb_nodes=self.nb_nodes,
            mems=self.mems)


    def parallel_process_output(self):
        for i in range(sample_times):
            device_choice = np.array(list(map(sample_func1, self.outputs[0:max_replica_num])))
            device_choice = np.transpose(device_choice)

            device_choice, replica_mask = post_process_device_choice(device_choice, self.batch_size)
            ps_or_reduce = np.array(list(map(random_func1, self.outputs[max_replica_num])))
            _reward, out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict,self.sink)
            if not out_of_memory:
                self.oom.append(False)
            else:
                self.oom.append(True)

            self.rewards.append(_reward)
            self.ps_or_reduces.append(ps_or_reduce)
            self.device_choices.append(device_choice)
            self.replica_masks.append(replica_mask)
    def post_parallel_process(self):
        for i in range(sample_times):
            self.cal_entropy = self.outputs[-1]
            if self.rewards[i] > self.best_reward:
                self.best_reward = self.rewards[i]
                self.best_replica_num = list()
                self.best_device_choice = self.device_choices[i]
                self.best_ps_or_reduce = self.ps_or_reduces[i]
            if not self.oom[i]:
                self.strategy_pool.insert(self.rewards[i], self.device_choices[i], self.replica_masks[i], self.ps_or_reduces[i])

    def process_output(self):
        for i in range(sample_times):
            replica_num = list()
            device_choice = np.array(list(map(sample_func1, self.outputs[0:max_replica_num])))
            device_choice = np.transpose(device_choice)

            device_choice, replica_mask = post_process_device_choice(device_choice, self.batch_size)
            ps_or_reduce = np.array(list(map(random_func1, self.outputs[max_replica_num])))

            self.cal_entropy = self.outputs[-1]
            _reward, out_of_memory = self.env.get_reward2(device_choice, ps_or_reduce, self.index_id_dict,self.sink)
            if _reward > self.best_reward:
                self.best_reward = _reward
                self.best_replica_num = replica_num
                self.best_device_choice = device_choice
                self.best_ps_or_reduce = ps_or_reduce
            if not out_of_memory:
                self.strategy_pool.insert(_reward, device_choice, replica_mask, ps_or_reduce)
                self.oom[i] = False

            self.rewards[i] = _reward
            self.ps_or_reduces[i] = ps_or_reduce
            self.device_choices[i] = device_choice
            self.replica_masks[i] = replica_mask
    def train(self,epoch):
        tr_step = 0
        co_entropy = 0
        tr_size = self.features.shape[0]
        if self.strategy_pool.get_length()>0:
            pool_strategy = self.strategy_pool.choose_strategy()
            self.rewards.append(pool_strategy["reward"])
            self.device_choices.append(pool_strategy["device_choice"])
            self.ps_or_reduces.append(np.reshape(pool_strategy["ps_or_reduce"],(self.nb_nodes,)))
            self.replica_masks.append(pool_strategy["replica_mask"])

        while tr_step * batch_size < tr_size:
            new_loss,self.mems=self.place_gnn.learn(ftr_in=self.features[tr_step * batch_size:(tr_step + 1) * batch_size],
                            bias_in=self.biases,
                            nb_nodes=self.nb_nodes,
                            replica_num_array=np.array(self.replica_masks),
                            sample_ps_or_reduce = np.array(self.ps_or_reduces),
                            sample_device_choice = np.array(self.device_choices),
                            time_ratio = (np.array(self.rewards)-self.avg)/np.abs(self.avg),
                            coef_entropy=co_entropy,
                            mems = self.mems)
            tr_step += 1
        for i in range(sample_times):
            if self.oom[i] == False:
                self.avg = (self.avg+self.rewards[i])/2
        times = self.rewards[0]*self.rewards[0]
        if epoch % show_interval == 0:
            print("[{}] step = {}".format(self.folder_path,epoch))
            print("[{}] time = {}".format(self.folder_path,times))
            print("[{}] average reward = {}".format(self.folder_path,np.mean(self.strategy_pool.rewards)))
            print("[{}] overall entropy:{}".format(self.folder_path,self.cal_entropy))
            with open(self.folder_path+"/time.log", "a+") as f:
                f.write(str(times) + ",")
            with open(self.folder_path+"/entropy.log", "a+") as f:
                f.write(str(self.cal_entropy) + ",")
            with open(self.folder_path+"/loss.log", "a+") as f:
                f.write(str(new_loss) + ",")

    def run(self):
        while True:
            self.event.wait()
            self.sample()
            self.process_output()
            self.event.clear()
            self.event2.set()


class new_place_GNN():
    def __init__(self,sess,ft_size):
        with tf.name_scope('place_gnn'):
            self.sess = sess

            self.ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, None, ft_size),name="ftr_in")
            #self.bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, None, None),name="bias_in")
            self.bias_in = tf.sparse_placeholder(dtype=tf.float32)
            self.nb_node = tf.placeholder(dtype=tf.int32, shape=(),name="nb_node")
            self.attn_drop = tf.placeholder(dtype=tf.float32, shape=(),name="attn_drop")
            self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=(),name="ffd_drop")
            self.is_train = tf.placeholder(dtype=tf.bool, shape=(),name="is_train")
            self.sample_ps_or_reduce = tf.placeholder(dtype=tf.int32, shape=(None,None,),name="sample_ps_or_reduce")
            self.sample_device_choice = tf.placeholder(dtype=tf.int32, shape=(None,None,max_replica_num,),name="sample_device_choice")
            self.replica_num_array = tf.placeholder(dtype=tf.float32, shape=(None,None,max_replica_num),name="replica_num_array")
            self.time_ratio = tf.placeholder(dtype=tf.float32, shape=(None,),name="time_ratio")
            self.coef_entropy = tf.placeholder(dtype=tf.float32, shape=(),name="coef_entropy")
            self.mems = [tf.placeholder(tf.float32,[2, None, 64]) for _ in range(3)]
        with tf.device("/device:GPU:1"):
            logits = model.inference(self.ftr_in, 1024, self.nb_node, self.is_train,
                                     self.attn_drop, self.ffd_drop,
                                     bias_mat=self.bias_in,
                                     hid_units=hid_units, n_heads=n_heads,
                                     residual=residual, activation=nonlinearity)

        if not transformer:
            log_resh = tf.reshape(logits, [-1, 1024])
            log_resh = tf.layers.dense(log_resh, units=1024, activation=tf.nn.relu)
            self.cell = tf.nn.rnn_cell.MultiRNNCell([self.get_a_cell() for _ in range(max_replica_num)])
            h0 = self.cell.zero_state(self.nb_node, np.float32)
            self.output,self.h = self.cell.call(log_resh, h0)
            self.ps_or_reduce = tf.layers.dense(self.output, units=2, activation=tf.nn.softmax)
            self.device_choices = list()
            sum=0
            out = tf.layers.dense(self.h[0].c, units=len(devices), activation=tf.nn.softmax)
            self.device_choices.append(out)
            sum = sum + tf.reduce_mean(tf.reduce_sum(tf.log(out + np.power(10.0, -9)) * out, 1))
            for i in range(max_replica_num-1):
                out = tf.layers.dense(self.h[i+1].c, units=len(devices)+1, activation=tf.nn.softmax)
                self.device_choices.append(out)
                sum = sum+tf.reduce_mean(tf.reduce_sum(tf.log(out + np.power(10.0, -9)) *out, 1))
            self.entropy = tf.reduce_sum(tf.log(self.ps_or_reduce + np.power(10.0, -9)) * self.ps_or_reduce, 1)
            self.entropy = -(tf.reduce_mean(self.entropy) + sum / max_replica_num)
        else:
            with tf.device("/device:GPU:0"):
                logits=model.inference(logits, max_replica_num*(len(devices)+1)+1, self.nb_node, self.is_train,
                                self.attn_drop, self.ffd_drop,
                                bias_mat=self.bias_in,
                                hid_units=place_hid_units, n_heads=place_n_heads,
                                residual=residual, activation=nonlinearity)

                self.device_choices = list()
                output = tf.reshape(logits, [-1, max_replica_num*(len(devices)+1)+1])
                output = tf.layers.dense(output, units=max_replica_num*(len(devices)+1)+1, activation=tf.nn.relu)
            '''
            for i in range(6):
                output = rel_multihead_attn(
                    w=log_resh,
                    r = self.bias_in,
                    d_model=max_replica_num*(len(devices)+1)+1,
                    mems=self.mems[i],
                    n_head=10,
                    d_head=50,
                    nb_nodes=self.nb_node)
            
            '''
            sum = 0
            o1 = tf.nn.softmax(output[:,0:len(devices)])
            self.device_choices.append(o1)
            sum = sum + tf.reduce_mean(tf.reduce_sum(tf.log(o1 + np.power(10.0, -9)) * o1, 1))
            for i in range(1,max_replica_num):
                oi = tf.nn.softmax(output[:,i*(len(devices)+1)-1:(i+1)*(len(devices)+1)-1])
                self.device_choices.append(oi)
                sum = sum + tf.reduce_mean(tf.reduce_sum(tf.log(oi + np.power(10.0, -9)) * oi, 1))
            self.ps_or_reduce = tf.nn.softmax(output[:,-2:])
            self.entropy = tf.reduce_sum(tf.log(self.ps_or_reduce + np.power(10.0, -9)) * self.ps_or_reduce, 1)
            self.entropy = -(tf.reduce_mean(self.entropy) + sum / max_replica_num)


        for i in range(sample_times+1):
            one_hot_sample = tf.one_hot(self.sample_ps_or_reduce[i], 2)
            print("one_hot_sample.shape")
            print(one_hot_sample.shape)

            prob = tf.reduce_sum( self.ps_or_reduce * one_hot_sample, 1)
            # prob = tf.reduce_prod(peirob)
            reward = tf.reduce_sum(tf.log(prob + np.power(10.0, -9)) * self.time_ratio[i])

            #first device choice n*m
            one_hot_sample = tf.one_hot(self.sample_device_choice[i][:, 0], len(devices))
            prob = tf.reduce_sum(self.device_choices[0] * one_hot_sample, 1) * self.replica_num_array[i][:, 0] + (
                        1 - self.replica_num_array[i][:, 0])
            reward += tf.reduce_sum(tf.log(prob + np.power(10.0, -9)) * self.time_ratio[i])

            #rest device choice n*(m+1)
            for j in range(1,max_replica_num):
                one_hot_sample = tf.one_hot(self.sample_device_choice[i][:,j], len(devices)+1)
                prob = tf.reduce_sum(self.device_choices[j] * one_hot_sample, 1) * self.replica_num_array[i][:,j]+(1-self.replica_num_array[i][:,j])
                reward+=tf.reduce_sum(tf.log(prob + np.power(10.0, -9)) * self.time_ratio[i])


        reward = reward/(sample_times+1) + self.coef_entropy * self.entropy
        self.loss = -reward
        self.train_op = model.training(self.loss, lr, l2_coef)

        '''
        for op in tf.get_default_graph().get_operations():
            if op.node_def.op in variable_ops:
                op._set_device("/device:CPU:0")

        '''

    def get_a_cell(self):
        return tf.nn.rnn_cell.BasicLSTMCell(num_units=64)

    def learn(self,ftr_in,bias_in,nb_nodes,replica_num_array,sample_ps_or_reduce,sample_device_choice,time_ratio,coef_entropy,mems):
        feed_dict = {}
        feed_dict[self.ftr_in]=ftr_in
        feed_dict[self.bias_in]=bias_in
        feed_dict[self.nb_node]=nb_nodes
        feed_dict[self.is_train]=True
        feed_dict[self.attn_drop]=0.1
        feed_dict[self.ffd_drop]=0.1
        feed_dict[self.replica_num_array]=replica_num_array
        feed_dict[self.sample_ps_or_reduce]=sample_ps_or_reduce
        feed_dict[self.sample_device_choice]=sample_device_choice
        feed_dict[self.time_ratio]=time_ratio
        feed_dict[self.coef_entropy]=coef_entropy
        for item1,item2 in zip(self.mems,mems):
            feed_dict[item1]=item2
        loss,mems,_ = self.sess.run([self.loss,self.mems,self.train_op],
                     feed_dict=feed_dict)
        return loss,mems
    def get_replica_num_prob_and_entropy(self,ftr_in,bias_in,nb_nodes,mems):
        fetch_list =[item for item in self.device_choices]
        fetch_list.append(self.ps_or_reduce)
        fetch_list.append(self.entropy)


        feed_dict = {}
        feed_dict[self.ftr_in]=ftr_in
        feed_dict[self.bias_in]=bias_in
        feed_dict[self.nb_node]=nb_nodes
        feed_dict[self.is_train]=True
        feed_dict[self.attn_drop]=0.1
        feed_dict[self.ffd_drop]=0.1
        for item1,item2 in zip(self.mems,mems):
            feed_dict[item1]=item2

        outputs = self.sess.run(fetch_list, feed_dict=feed_dict)
        return outputs


def architecture_three():
    global  global_pool,models
    for i,feature_folder in enumerate(feature_folders):
        event = threading.Event()
        event2 = threading.Event()

        item = feature_item(feature_folder,global_pool,event,event2,sinks[i])
        item.setDaemon(True)
        models.append(item)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
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
                #model.sample_one_time()
                #model.wait_sample()
                model.sync_sample_and_parallel_process()
                model.post_parallel_process()
                model.train(epoch)

            '''
            for model in models:
                model.sample_one_time()

            for model in models:
                model.wait_sample()
            for model in models:
                model.post_parallel_process()

            for model in models:
                model.train(epoch)
            '''
            if epoch % show_interval == 0:
                saver.save(sess, checkpt_file)

        sess.close()

import signal
def handler(signum, frame):
    global  global_pool,models
    try:
        for model in models:
            model.proc.terminate()
    except:
        pass
    sys.exit()

if __name__ == '__main__':

    models = list()
    global_pool = Pool(1)
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)

    architecture_three()
