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
import os
import pickle
sys.path.append('../')
import tge
import json


class Environment(object):
    def __init__(self,gdef_path,devices):

        self.gdef = graph_pb2.GraphDef()
        with open(gdef_path,"r")as f:
            txt = f.read()
        pbtf.Parse(txt,self.gdef)
        self.random_strategy=list()

        self.best_reward=sys.maxsize
        self.best_strategy = dict()


        random_strategies_file = "data/graph/random_strategies.pkl"
        if os.path.exists(random_strategies_file):
            with open(random_strategies_file,"rb") as fh:
                self.random_strategy = pickle.load(fh)
        else:
            random_time = min(2**len(devices),2**8)
            while random_time>0:
                stratey = list()
                for j in range(len(devices)+1):
                    item = np.random.choice(2, p=[0.5,0.5])
                    stratey.append(int(item))
                if sum(stratey[1:])==0:
                    continue
                else:
                    self.random_strategy.append(stratey)
                    random_time = random_time-1
            with open(random_strategies_file,"wb") as fh:
                pickle.dump(self.random_strategy,fh)


        self.strategy_reward_dict=dict()
        self.devices =devices
        self.name_cost_dict = self.get_name_cost_dict()
        self._tge = tge.TGE(self.gdef, devices)

    def get_reward2(self,strategy,device_choice,ps_or_reduce,index_id_dict):

        new_device_array = np.zeros(shape=device_choice.shape,dtype=np.int32)
        for i in range(device_choice.shape[0]):
            for j in range(device_choice.shape[1]):
                if device_choice[i,j]!=-1 and device_choice[i,j]!=len(self.devices):
                    new_device_array[i,device_choice[i,j]]+=1
        ps_or_reduce = np.reshape(ps_or_reduce, (ps_or_reduce.shape[0], 1))
        new_device_array = np.concatenate((ps_or_reduce,new_device_array),axis=1)
        strategy = {index_id_dict[index]:new_device_array[index].tolist() for index in range(new_device_array.shape[0])}
        reward = tge.TGE(copy.deepcopy(self.gdef), self.devices).custom(strategy).evaluate(self.name_cost_dict)
        #reward = np.sum(strategy*strategy)
        if reward<self.best_reward:
            self.best_reward = reward
            self.best_strategy = strategy
            with open("best_reward.log", "w") as f:
                tmp = dict()
                tmp["time"] = reward
                tmp["strategy"] = self.best_strategy
                tmp["cost"] = self.name_cost_dict
                json.dump(tmp, f)

        return np.float32(reward)

    def get_reward(self,strategy,index_id_dict):
        if self.strategy_reward_dict.get(str(strategy),None):
            reward= self.strategy_reward_dict.get(str(strategy))
        else:
            reward = tge.TGE(copy.deepcopy(self.gdef), self.devices).custom({index_id_dict[index]:self.random_strategy[strategy_int] for index,strategy_int in enumerate(strategy)}).evaluate(self.name_cost_dict)
            #reward = np.sum(strategy*strategy)
            self.strategy_reward_dict[str(strategy)]=reward
        return np.float32(reward)

    def get_name_cost_dict(self):
        name_cost_dict = dict()
        with open("data/graph/docs.txt", "r") as f:
            for line in f.readlines():
                line = line.strip()
                items = line.split(" ")
                name = items[0]
                cost = items[1]
                name_cost_dict[name] = int(float(cost))
        for key,value in name_cost_dict.items():
            name_cost_dict[key] = [value]*len(self.devices)
        return name_cost_dict

checkpt_file = 'pre_trained/cora/mod_cora.ckpt'

dataset = 'cora'

# training params
batch_size = 1
nb_epochs = 100000
patience = 100
lr = 0.01  # learning rate
l2_coef = 0.0005  # weight decay
hid_units = [8] # numbers of hidden units per each attention head in each layer
n_heads = [8, 1] # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = GAT

print('Dataset: ' + dataset)
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


dataset = load_cora("data/graph",NewWhiteSpaceTokenizer())


#adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = process.load_data(dataset)
#features, spars = process.preprocess_features(features)

adj = dataset.adj_matrix(sparse=True)
feature_matrix, feature_masks = dataset.feature_matrix(bag_of_words=False, sparse=False)




labels, label_masks = dataset.label_list_or_matrix(one_hot=False)

train_node_indices, test_node_indices, train_masks, test_masks = dataset.split_train_and_test(training_rate=0.8)

n_values = np.max(labels) + 1
labels=np.eye(n_values)[labels]


nb_nodes = feature_matrix.shape[0]
ft_size = feature_matrix.shape[1]
nb_classes = 6

adj = adj.todense()

y_train = labels
y_val = labels
y_test = labels
train_mask=train_masks
val_mask=test_masks
test_mask=test_masks

index_id_dict = dataset.network.get_indexer(N_TYPE_NODE).index_id_dict
features = feature_matrix[np.newaxis]
adj = adj[np.newaxis]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases = process.adj_to_bias(adj, [nb_nodes], nhood=5)

devices = (
    "/job:tge/replica:0/task:0/device:GPU:0",
    "/job:tge/replica:0/task:0/device:GPU:1",
    "/job:tge/replica:0/task:1/device:GPU:0",
    "/job:tge/replica:0/task:1/device:GPU:1"
)
env = Environment("data/graph/graph.pbtxt",devices)

show_interval=10



class Stratey_one_GNN():
    def __init__(self,sess):
        self.sess = sess
        with tf.name_scope('Stratey_one'):
            self.ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
            self.bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
            self.msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
            self.attn_drop = tf.placeholder(dtype=tf.float32, shape=())
            self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
            self.is_train = tf.placeholder(dtype=tf.bool, shape=())
            self.sample_strategy = tf.placeholder(dtype=tf.int32, shape=(nb_nodes,))
            self.step_reward = tf.placeholder(dtype=tf.float32, shape=())
            self.avg_reward = tf.placeholder(dtype=tf.float32, shape=())
            self.coef_entropy = tf.placeholder(dtype=tf.float32, shape=())
            logits = model.inference(self.ftr_in, nb_classes, nb_nodes, self.is_train,
                                     self.attn_drop, self.ffd_drop,
                                     bias_mat=self.bias_in,
                                     hid_units=hid_units, n_heads=n_heads,
                                     residual=residual, activation=nonlinearity)
            log_resh = tf.reshape(logits, [-1, nb_classes])
            msk_resh = tf.reshape(self.msk_in, [-1])

            self.preds = tf.nn.softmax(tf.nn.l2_normalize(log_resh,axis = -1))

            self.entropy = tf.reduce_sum(tf.log(self.preds + np.power(10.0, -9)) * self.preds, 1)
            self.entropy = tf.reduce_mean(self.entropy)

            print(logits.shape)
            print(log_resh.shape)
            print(self.preds.shape)
            print("sample_strategy.shape")
            print(self.sample_strategy.shape)
            one_hot_sample = tf.one_hot(self.sample_strategy, nb_classes)
            print("one_hot_sample.shape")
            print(one_hot_sample.shape)

            prob = tf.reduce_sum(self.preds * one_hot_sample, 1)
            # prob = tf.reduce_prod(peirob)
            reward = tf.reduce_sum(tf.log(prob + np.power(10.0, -9)) * (self.step_reward - self.avg_reward)/100000)
            self.loss = reward# + self.coef_entropy * self.entropy
            self.train_op = model.training(self.loss, lr, l2_coef)

    def learn(self,ftr_in,bias_in,msk_in,sample_strategy,step_reward,avg_reward,coef_entropy):
        loss,_ = self.sess.run([self.loss,self.train_op],
                     feed_dict={
                         self.ftr_in: ftr_in,
                         self.bias_in:bias_in,
                         self.msk_in: msk_in,
                         self.is_train: True,
                         self.attn_drop: 0.6, self.ffd_drop: 0.6, self.sample_strategy: sample_strategy,
                         self.step_reward: step_reward, self.avg_reward: avg_reward, self.coef_entropy: coef_entropy})
        return loss
    def get_replica_num_prob_and_entropy(self,ftr_in,bias_in,msk_in):
        cal_preds, cal_entropy = self.sess.run([self.preds, self.entropy], feed_dict={
            self.ftr_in: ftr_in,
            self.bias_in: bias_in,
            self.msk_in: msk_in,
            self.is_train: True,
            self.attn_drop: 0.6, self.ffd_drop: 0.6})
        return cal_preds,cal_entropy



class replica_GNN():
    def __init__(self,sess):
        with tf.name_scope('relica_gnn'):
            self.sess = sess

            self.ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size))
            self.bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes))
            self.msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes))
            self.attn_drop = tf.placeholder(dtype=tf.float32, shape=())
            self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=())
            self.is_train = tf.placeholder(dtype=tf.bool, shape=())
            self.sample_relica_num = tf.placeholder(dtype=tf.int32, shape=(nb_nodes,))
            self.step_reward = tf.placeholder(dtype=tf.float32, shape=())
            self.avg_reward = tf.placeholder(dtype=tf.float32, shape=())
            self.coef_entropy = tf.placeholder(dtype=tf.float32, shape=())

        logits = model.inference(self.ftr_in, len(devices), nb_nodes, self.is_train,
                                 self.attn_drop, self.ffd_drop,
                                 bias_mat=self.bias_in,
                                 hid_units=hid_units, n_heads=n_heads,
                                 residual=residual, activation=nonlinearity)
        log_resh = tf.reshape(logits, [-1, len(devices)])
        msk_resh = tf.reshape(self.msk_in, [-1])

        self.preds = tf.nn.softmax(tf.nn.l2_normalize(log_resh,axis = -1))
        self.entropy = tf.reduce_sum(tf.log(self.preds + np.power(10.0, -9)) * self.preds, 1)
        self.entropy = tf.reduce_mean(self.entropy)

        print(logits.shape)
        print(log_resh.shape)
        print(self.preds.shape)
        print("sample_relica_num.shape")
        print(self.sample_relica_num.shape)
        one_hot_sample = tf.one_hot(self.sample_relica_num, len(devices))
        print("one_hot_sample.shape")
        print(one_hot_sample.shape)

        prob = tf.reduce_sum(self.preds * one_hot_sample, 1)
        # prob = tf.reduce_prod(peirob)
        reward = tf.reduce_sum(tf.log(prob + np.power(10.0, -9)) * (self.step_reward - self.avg_reward)/100000)
        self.loss = reward# + self.coef_entropy * self.entropy
        self.train_op = model.training(self.loss, lr, l2_coef)

    def learn(self,ftr_in,bias_in,msk_in,sample_relica_num,step_reward,avg_reward,coef_entropy):
        loss,_ = self.sess.run([self.loss,self.train_op],
                     feed_dict={
                         self.ftr_in: ftr_in,
                         self.bias_in:bias_in,
                         self.msk_in: msk_in,
                         self.is_train: True,
                         self.attn_drop: 0.6, self.ffd_drop: 0.6, self.sample_relica_num: sample_relica_num,
                         self.step_reward: step_reward, self.avg_reward: avg_reward, self.coef_entropy: coef_entropy})
        return loss
    def get_replica_num_prob_and_entropy(self,ftr_in,bias_in,msk_in):
        cal_preds, cal_entropy = self.sess.run([self.preds, self.entropy], feed_dict={
            self.ftr_in: ftr_in,
            self.bias_in: bias_in,
            self.msk_in: msk_in,
            self.is_train: True,
            self.attn_drop: 0.6, self.ffd_drop: 0.6})
        return cal_preds,cal_entropy


class place_GNN():
    def __init__(self,sess):
        with tf.name_scope('place_gnn'):
            self.sess = sess

            self.ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, ft_size+1),name="ftr_in")
            self.bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, nb_nodes, nb_nodes),name="bias_in")
            self.msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes),name="msk_in")
            self.attn_drop = tf.placeholder(dtype=tf.float32, shape=(),name="attn_drop")
            self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=(),name="ffd_drop")
            self.is_train = tf.placeholder(dtype=tf.bool, shape=(),name="is_train")
            self.sample_ps_or_reduce = tf.placeholder(dtype=tf.int32, shape=(nb_nodes,),name="sample_ps_or_reduce")
            self.sample_device_choice = tf.placeholder(dtype=tf.int32, shape=(nb_nodes,len(devices),),name="sample_device_choice")
            self.replica_num_array = tf.placeholder(dtype=tf.float32, shape=(nb_nodes,len(devices)),name="replica_num_array")
            self.step_reward = tf.placeholder(dtype=tf.float32, shape=(),name="step_reward")
            self.avg_reward = tf.placeholder(dtype=tf.float32, shape=(),name="avg_reward")
            self.coef_entropy = tf.placeholder(dtype=tf.float32, shape=(),name="coef_entropy")

        logits = model.inference(self.ftr_in, nb_classes, nb_nodes, self.is_train,
                                 self.attn_drop, self.ffd_drop,
                                 bias_mat=self.bias_in,
                                 hid_units=hid_units, n_heads=n_heads,
                                 residual=residual, activation=nonlinearity)
        log_resh = tf.reshape(logits, [-1, nb_classes])
        msk_resh = tf.reshape(self.msk_in, [-1])

        self.cell = tf.nn.rnn_cell.MultiRNNCell([self.get_a_cell() for _ in range(len(devices))])
        #inputs = tf.placeholder(np.float32, shape=(nb_nodes, nb_classes))
        h0 = self.cell.zero_state(nb_nodes, np.float32)
        self.output,self.h = self.cell.call(log_resh, h0)
        self.ps_or_reduce = tf.layers.dense(self.output, units=2, activation=tf.nn.relu)
        self.ps_or_reduce = tf.nn.softmax(tf.nn.l2_normalize(self.ps_or_reduce,axis = -1))
        self.device_choices = list()
        sum=0
        for i in range(len(devices)):
            out = tf.nn.softmax(tf.nn.l2_normalize(self.h[i].c,axis = -1))
            self.device_choices.append(out)
            sum = sum+tf.reduce_mean(tf.reduce_sum(tf.log(out + np.power(10.0, -9)) *out, 1))

        self.entropy = tf.reduce_sum(tf.log(self.ps_or_reduce + np.power(10.0, -9)) * self.ps_or_reduce, 1)
        self.entropy = tf.reduce_mean(self.entropy)+sum/len(devices)

        one_hot_sample = tf.one_hot(self.sample_ps_or_reduce, 2)
        print("one_hot_sample.shape")
        print(one_hot_sample.shape)

        prob = tf.reduce_sum( self.ps_or_reduce * one_hot_sample, 1)
        # prob = tf.reduce_prod(peirob)
        reward = tf.reduce_sum(tf.log(prob + np.power(10.0, -9)) * (self.step_reward - self.avg_reward))

        for i in range(len(devices)):
            one_hot_sample = tf.one_hot(self.sample_device_choice[:,i], len(devices))
            prob = tf.reduce_sum(self.device_choices[i] * one_hot_sample, 1) * self.replica_num_array[:,i]+(1-self.replica_num_array[:,i])
            reward+=tf.reduce_sum(tf.log(prob + np.power(10.0, -9)) * (self.step_reward - self.avg_reward)/100000)


        self.loss = reward #+ self.coef_entropy * self.entropy
        self.train_op = model.training(self.loss, lr, l2_coef)



    def get_a_cell(self):
        return tf.nn.rnn_cell.BasicLSTMCell(num_units=len(devices))

    def learn(self,ftr_in,bias_in,msk_in,replica_num_array,sample_ps_or_reduce,sample_device_choice,step_reward,avg_reward,coef_entropy):
        loss,_ = self.sess.run([self.loss,self.train_op],
                     feed_dict={
                         self.ftr_in: ftr_in,
                         self.bias_in:bias_in,
                         self.msk_in: msk_in,
                         self.is_train: True,
                         self.attn_drop: 0.6, self.ffd_drop: 0.6, self.replica_num_array: replica_num_array,
                         self.sample_ps_or_reduce:sample_ps_or_reduce,
                         self.sample_device_choice:sample_device_choice,
                         self.step_reward: step_reward, self.avg_reward: avg_reward, self.coef_entropy: coef_entropy})
        return loss
    def get_replica_num_prob_and_entropy(self,ftr_in,bias_in,msk_in):
        fetch_list =[item for item in self.device_choices]
        fetch_list.append(self.ps_or_reduce)
        fetch_list.append(self.entropy)
        outputs = self.sess.run(fetch_list, feed_dict={
            self.ftr_in: ftr_in,
            self.bias_in: bias_in,
            self.msk_in: msk_in,
            self.is_train: True,
            self.attn_drop: 0.6, self.ffd_drop: 0.6})
        return outputs


class new_place_GNN():
    def __init__(self,sess):
        with tf.name_scope('place_gnn'):
            self.sess = sess

            self.ftr_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, None, ft_size),name="ftr_in")
            self.bias_in = tf.placeholder(dtype=tf.float32, shape=(batch_size, None, None),name="bias_in")
            self.nb_node = tf.placeholder(dtype=tf.int32, shape=(),name="nb_node")
            self.attn_drop = tf.placeholder(dtype=tf.float32, shape=(),name="attn_drop")
            self.ffd_drop = tf.placeholder(dtype=tf.float32, shape=(),name="ffd_drop")
            self.is_train = tf.placeholder(dtype=tf.bool, shape=(),name="is_train")
            self.sample_ps_or_reduce = tf.placeholder(dtype=tf.int32, shape=(None,),name="sample_ps_or_reduce")
            self.sample_device_choice = tf.placeholder(dtype=tf.int32, shape=(None,len(devices),),name="sample_device_choice")
            self.replica_num_array = tf.placeholder(dtype=tf.float32, shape=(None,len(devices)),name="replica_num_array")
            self.step_reward = tf.placeholder(dtype=tf.float32, shape=(),name="step_reward")
            self.avg_reward = tf.placeholder(dtype=tf.float32, shape=(),name="avg_reward")
            self.coef_entropy = tf.placeholder(dtype=tf.float32, shape=(),name="coef_entropy")

        logits = model.inference(self.ftr_in, nb_classes, nb_nodes, self.is_train,
                                 self.attn_drop, self.ffd_drop,
                                 bias_mat=self.bias_in,
                                 hid_units=hid_units, n_heads=n_heads,
                                 residual=residual, activation=nonlinearity)
        log_resh = tf.reshape(logits, [-1, nb_classes])

        self.cell = tf.nn.rnn_cell.MultiRNNCell([self.get_a_cell() for _ in range(len(devices))])
        #inputs = tf.placeholder(np.float32, shape=(nb_nodes, nb_classes))
        h0 = self.cell.zero_state(self.nb_node, np.float32)
        self.output,self.h = self.cell.call(log_resh, h0)
        self.ps_or_reduce = tf.layers.dense(self.output, units=2, activation=tf.nn.softmax)
        #self.ps_or_reduce = tf.nn.softmax(tf.nn.l2_normalize(self.ps_or_reduce,axis = -1))
        self.device_choices = list()
        sum=0
        out = tf.layers.dense(self.h[0].c, units=len(devices), activation=tf.nn.softmax)
        self.device_choices.append(out)
        sum = sum + tf.reduce_mean(tf.reduce_sum(tf.log(out + np.power(10.0, -9)) * out, 1))
        for i in range(len(devices)-1):
            out = tf.layers.dense(self.h[i+1].c, units=len(devices)+1, activation=tf.nn.softmax)
            self.device_choices.append(out)
            sum = sum+tf.reduce_mean(tf.reduce_sum(tf.log(out + np.power(10.0, -9)) *out, 1))

        self.entropy = tf.reduce_sum(tf.log(self.ps_or_reduce + np.power(10.0, -9)) * self.ps_or_reduce, 1)
        self.entropy = tf.reduce_mean(self.entropy)+sum/len(devices)

        one_hot_sample = tf.one_hot(self.sample_ps_or_reduce, 2)
        print("one_hot_sample.shape")
        print(one_hot_sample.shape)

        prob = tf.reduce_sum( self.ps_or_reduce * one_hot_sample, 1)
        # prob = tf.reduce_prod(peirob)
        reward = tf.reduce_sum(tf.log(prob + np.power(10.0, -9)) * (self.step_reward - self.avg_reward))

        #first device choice n*m
        one_hot_sample = tf.one_hot(self.sample_device_choice[:, 0], len(devices))
        prob = tf.reduce_sum(self.device_choices[0] * one_hot_sample, 1) * self.replica_num_array[:, 0] + (
                    1 - self.replica_num_array[:, 0])
        reward += tf.reduce_sum(tf.log(prob + np.power(10.0, -9)) * (self.step_reward - self.avg_reward) / 100000)

        #rest device choice n*(m+1)
        for i in range(1,len(devices)):
            one_hot_sample = tf.one_hot(self.sample_device_choice[:,i], len(devices)+1)
            prob = tf.reduce_sum(self.device_choices[i] * one_hot_sample, 1) * self.replica_num_array[:,i]+(1-self.replica_num_array[:,i])
            reward+=tf.reduce_sum(tf.log(prob + np.power(10.0, -9)) * (self.step_reward - self.avg_reward)/100000)


        self.loss = reward #+ self.coef_entropy * self.entropy
        self.train_op = model.training(self.loss, lr, l2_coef)



    def get_a_cell(self):
        return tf.nn.rnn_cell.BasicLSTMCell(num_units=256)

    def learn(self,ftr_in,bias_in,nb_nodes,replica_num_array,sample_ps_or_reduce,sample_device_choice,step_reward,avg_reward,coef_entropy):
        loss,_ = self.sess.run([self.loss,self.train_op],
                     feed_dict={
                         self.ftr_in: ftr_in,
                         self.bias_in:bias_in,
                         self.nb_node:nb_nodes,
                         self.is_train: True,
                         self.attn_drop: 0.6, self.ffd_drop: 0.6, self.replica_num_array: replica_num_array,
                         self.sample_ps_or_reduce:sample_ps_or_reduce,
                         self.sample_device_choice:sample_device_choice,
                         self.step_reward: step_reward, self.avg_reward: avg_reward, self.coef_entropy: coef_entropy})
        return loss
    def get_replica_num_prob_and_entropy(self,ftr_in,bias_in,nb_nodes):
        fetch_list =[item for item in self.device_choices]
        fetch_list.append(self.ps_or_reduce)
        fetch_list.append(self.entropy)
        outputs = self.sess.run(fetch_list, feed_dict={
            self.ftr_in: ftr_in,
            self.bias_in: bias_in,
            self.nb_node:nb_nodes,
            self.is_train: True,
            self.attn_drop: 0.6, self.ffd_drop: 0.6})
        return outputs

def architecture_three():
    with tf.Session() as sess:
        place_gnn = new_place_GNN(sess)
        saver = tf.train.Saver()
        average_reward=0
        try:
            saver.restore(sess, checkpt_file)
        except:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

        co_entropy = 1000
        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = features.shape[0]
            replica_nums = list()
            replica_n_hot_nums = list()
            device_choices = list()
            rewards = list()
            ps_or_reduces=list()
            new_features=list()
            if epoch % 100 == 0:
                co_entropy = co_entropy * 0.95

            for i in range(10):
                replica_num = list()
                replica_n_hot_num = np.zeros(shape=(nb_nodes,len(devices)),dtype=np.int32)
                device_choice = np.zeros(shape=(nb_nodes,len(devices)),dtype=np.int32)
                ps_or_reduce=list()
                outputs = place_gnn.get_replica_num_prob_and_entropy(ftr_in=features[tr_step * batch_size:(tr_step + 1) * batch_size],
                                                                          bias_in=biases[tr_step * batch_size:(tr_step + 1) * batch_size],
                                                                     nb_nodes=nb_nodes)

                finished_node = list()
                for i in range(len(devices)):
                    output = outputs[i]
                    for j,pred in enumerate(output):
                        if j not in finished_node:
                            index = np.random.choice(pred.size,p=pred)
                            if index==len(devices):
                                finished_node.append(j)
                                device_choice[j, i] = index
                                replica_n_hot_num[j,i]=1
                            else:
                                device_choice[j,i] = index
                                replica_n_hot_num[j,i]=1
                        else:
                            device_choice[j,i] = -1
                device_choices.append(device_choice)
                replica_n_hot_nums.append(replica_n_hot_num)

                for j, pred in enumerate(outputs[len(devices)]):
                    index = np.random.choice(pred.size, p=pred)
                    ps_or_reduce.append(index)
                ps_or_reduce = np.array(ps_or_reduce)
                ps_or_reduces.append(ps_or_reduce)

                cal_entropy = outputs[-1]
                _reward = env.get_reward2(replica_num, device_choice,ps_or_reduce,index_id_dict)
                rewards.append(_reward)
            average_reward = average_reward*0.1+0.9*np.mean(rewards)

            for i in range(10):
                while tr_step * batch_size < tr_size:
                    new_loss=place_gnn.learn(ftr_in=features[tr_step * batch_size:(tr_step + 1) * batch_size],
                                    bias_in=biases[tr_step * batch_size:(tr_step + 1) * batch_size],
                                    nb_nodes=nb_nodes,
                                    replica_num_array=replica_n_hot_nums[i],
                                    sample_ps_or_reduce = ps_or_reduces[i],
                                    sample_device_choice = device_choices[i],
                                    step_reward=rewards[i],
                                    avg_reward=average_reward,
                                    coef_entropy=co_entropy)
                    tr_step += 1
                tr_step = 0


            if epoch % show_interval == 0:
                print("step = {}".format(epoch))
                print("reward = {}".format(rewards))
                print("average reward = {}".format(np.mean(rewards)))
                print("overall entropy:{}".format(cal_entropy))
                with open("reward_log.log", "a+") as f:
                    f.write(str(np.mean(rewards)) + ",")
                with open("entropy.log", "a+") as f:
                    f.write(str(cal_entropy) + ",")
                with open("loss.log", "a+") as f:
                    f.write(str(new_loss) + ",")

                saver.save(sess, checkpt_file)

        sess.close()

def architecture_two():
    with tf.Session() as sess:
        replica_gnn = replica_GNN(sess)
        place_gnn = place_GNN(sess)
        saver = tf.train.Saver()
        global_rewards = list()
        try:
            saver.restore(sess, checkpt_file)
        except:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)


        co_entropy = 1000
        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = features.shape[0]
            replica_nums = list()
            replica_n_hot_nums = list()
            device_choices = list()
            rewards = list()
            ps_or_reduces=list()
            new_features=list()
            if epoch % 100 == 0:
                co_entropy = co_entropy * 0.95
            cal_preds, cal_entropy = replica_gnn.get_replica_num_prob_and_entropy(ftr_in=features[tr_step * batch_size:(tr_step + 1) * batch_size],
                                                                          bias_in=biases[tr_step * batch_size:(tr_step + 1) * batch_size],
                                                                          msk_in=train_mask[tr_step * batch_size:(tr_step + 1) * batch_size])
            for i in range(10):
                replica_num = list()
                replica_n_hot_num = list()
                device_choice = np.zeros(shape=(nb_nodes,len(devices)),dtype=np.int32)
                ps_or_reduce=list()
                for pred in cal_preds:
                    item = np.random.choice(pred.size, p=pred)+1
                    n_hot_item =np.array( [1 if a<item else 0 for a in range(len(devices))])
                    replica_num.append(item)
                    replica_n_hot_num.append(n_hot_item)
                replica_num = np.array(replica_num)
                replica_n_hot_num = np.array(replica_n_hot_num)
                # strategy = np.array([np.random.choice(pred.size, p=pred) for pred in cal_preds])
                replica_nums.append(replica_num)
                replica_n_hot_nums.append(replica_n_hot_num)

                new_replica_num = np.reshape(replica_num,(nb_nodes,1))
                new_feature = np.concatenate((features[tr_step * batch_size:(tr_step + 1) * batch_size][0], new_replica_num), axis=1)
                new_feature = new_feature[np.newaxis]
                new_features.append(new_feature)
                outputs = place_gnn.get_replica_num_prob_and_entropy(ftr_in=new_feature,
                                                                          bias_in=biases[tr_step * batch_size:(tr_step + 1) * batch_size],
                                                                          msk_in=train_mask[tr_step * batch_size:(tr_step + 1) * batch_size])

                for i in range(len(devices)):
                    output = outputs[i]
                    for j,pred in enumerate(output):
                        if i<replica_num[j]:
                            index = np.random.choice(pred.size,p=pred)
                            device_choice[j,i] = index
                        else:
                            device_choice[j,i] = -1
                device_choices.append(device_choice)

                for j, pred in enumerate(outputs[len(devices)]):
                    index = np.random.choice(pred.size, p=pred)
                    ps_or_reduce.append(index)
                ps_or_reduce = np.array(ps_or_reduce)
                ps_or_reduces.append(ps_or_reduce)

                cal_entropy = outputs[-1]

                rewards.append(env.get_reward2(replica_num, device_choice,ps_or_reduce,index_id_dict))
            global_rewards.extend(rewards)
            average_reward = np.float32(sum(global_rewards) / len(global_rewards))

            for i in range(10):
                while tr_step * batch_size < tr_size:
                    new_loss=place_gnn.learn(ftr_in=new_features[i],
                                    bias_in=biases[tr_step * batch_size:(tr_step + 1) * batch_size],
                                    msk_in=train_mask[tr_step * batch_size:(tr_step + 1) * batch_size],
                                    replica_num_array=replica_n_hot_nums[i],
                                    sample_ps_or_reduce = ps_or_reduces[i],
                                    sample_device_choice = device_choices[i],
                                    step_reward=rewards[i],
                                    avg_reward=average_reward,
                                    coef_entropy=co_entropy)
                    tr_step += 1
                tr_step = 0

                while tr_step * batch_size < tr_size:
                    replica_gnn.learn(ftr_in=features[tr_step * batch_size:(tr_step + 1) * batch_size],
                                        bias_in=biases[tr_step * batch_size:(tr_step + 1) * batch_size],
                                        msk_in=train_mask[tr_step * batch_size:(tr_step + 1) * batch_size],
                                        sample_relica_num=replica_nums[i],
                                        step_reward=rewards[i],
                                        avg_reward=average_reward,
                                        coef_entropy=co_entropy)
                    tr_step += 1
                tr_step = 0

            if epoch % show_interval == 0:
                print("step = {}".format(epoch))
                print("reward = {}".format(rewards))
                print("average reward = {}".format(np.mean(rewards)))
                print("overall entropy:{}".format(cal_entropy))
                with open("reward_log.log", "a+") as f:
                    f.write(str(np.mean(rewards)) + ",")
                with open("entropy.log", "a+") as f:
                    f.write(str(cal_entropy) + ",")
                with open("loss.log", "a+") as f:
                    f.write(str(new_loss) + ",")

                saver.save(sess, checkpt_file)

        sess.close()
def architecture_one():
    with tf.Session() as sess:
        gnn = Stratey_one_GNN(sess)
        saver = tf.train.Saver()
        global_rewards = list()

        try:
            saver.restore(sess, checkpt_file)
        except:
            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)

        co_entropy = 1000
        best_strategy=dict()
        best_reward=sys.maxsize
        for epoch in range(nb_epochs):
            tr_step = 0
            tr_size = features.shape[0]
            sample_strategies = list()
            rewards = list()
            if epoch % 100 == 0:
                co_entropy = co_entropy * 0.95
            cal_preds, cal_entropy = gnn.get_replica_num_prob_and_entropy(ftr_in=features[tr_step * batch_size:(tr_step + 1) * batch_size],
                                                                          bias_in=biases[tr_step * batch_size:(tr_step + 1) * batch_size],
                                                                          msk_in=train_mask[tr_step * batch_size:(tr_step + 1) * batch_size])
            for i in range(10):
                strategy = list()
                for pred in cal_preds:
                    item = np.random.choice(pred.size, p=pred)
                    strategy.append(item)
                strategy = np.array(strategy)
                # strategy = np.array([np.random.choice(pred.size, p=pred) for pred in cal_preds])
                sample_strategies.append(strategy)
                _reward = env.get_reward(strategy, index_id_dict)
                rewards.append(_reward)
                if _reward<best_reward:
                    best_strategy = strategy
                    best_reward = _reward
            global_rewards.extend(rewards)
            average_reward = np.float32(sum(global_rewards) / len(global_rewards))

            for i in range(10):
                while tr_step * batch_size < tr_size:
                    new_loss=gnn.learn(ftr_in=features[tr_step * batch_size:(tr_step + 1) * batch_size],
                              bias_in=biases[tr_step * batch_size:(tr_step + 1) * batch_size],
                              msk_in=train_mask[tr_step * batch_size:(tr_step + 1) * batch_size],
                              sample_strategy=sample_strategies[i],
                              step_reward=rewards[i],
                              avg_reward=average_reward,
                              coef_entropy=co_entropy)
                    tr_step += 1
                tr_step = 0

            if epoch % show_interval == 0:
                print("step = {}".format(epoch))
                print("reward = {}".format(rewards))
                print("average reward = {}".format(np.mean(rewards)))
                print("overall entropy:{}".format(cal_entropy))
                with open("reward_log.log", "a+") as f:
                    f.write(str(np.mean(rewards)) + ",")
                with open("entropy.log", "a+") as f:
                    f.write(str(cal_entropy) + ",")
                with open("loss.log", "a+") as f:
                    f.write(str(new_loss) + ",")


                saver.save(sess, checkpt_file)


        sess.close()


architecture_three()