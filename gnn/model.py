import tensorflow as tf
import dgl.function as fn
import numpy as np
from utils import info

all_etypes = ["link", "prev", "succ", "place", "serve"]

class GConv(tf.keras.layers.Layer):
    '''Graph Conv layer that concats the edge features before sending message'''
    def __init__(self, out_feats, activation=None):
        super(GConv, self).__init__()
        self.layers = { etype: tf.keras.layers.Dense(out_feats, activation=activation) for etype in all_etypes }
        self.op_final = tf.keras.layers.Dense(out_feats, activation=None)
        self.device_final = tf.keras.layers.Dense(out_feats, activation=None)

    def call(self, graph, op_feats, device_feats, edge_feats):
        op_dst, device_dst = [], []
        for stype, etype, dtype in graph.canonical_etypes:
            g = graph[etype].local_var()
            if stype == 'op':
                g.srcdata['h'] = op_feats
            elif stype == 'device':
                g.srcdata['h'] = device_feats

            g.apply_edges(fn.copy_u('h', 's'))
            edata = tf.concat([g.edata.pop('s'), edge_feats[etype]], axis=1)
            g.edata['e'] = self.layers[etype](edata)
            g.update_all(fn.copy_e('e', 'm'), fn.sum(msg='m', out='h'))

            if dtype == 'op':
                op_dst.append(g.dstdata['h'])
            elif dtype == 'device':
                device_dst.append(g.dstdata['h'])

        op_dst = self.op_final(tf.math.add_n(op_dst))
        device_dst = self.device_final(tf.math.add_n(device_dst))

        return op_feats + op_dst, device_feats + device_dst

class Model(tf.keras.Model):
    def __init__(self, op_table):
        super(Model, self).__init__()

        node_hidden = 64
        edge_hidden = 8
        op_embedding_len = 8

        self.op_embedding = tf.keras.layers.Embedding(len(op_table), op_embedding_len, input_length=1)

        self.op_trans = tf.keras.layers.Dense(node_hidden, activation=tf.math.tanh)
        self.device_trans = tf.keras.layers.Dense(node_hidden, activation=tf.math.tanh)
        self.edge_trans = { etype: tf.keras.layers.Dense(edge_hidden, activation=tf.math.tanh) for etype in all_etypes }

        self.gconv_layers = [
            GConv(node_hidden, tf.math.tanh),
            GConv(node_hidden, tf.math.tanh),
            GConv(node_hidden, tf.math.tanh)
        ]

        self.final_place = tf.keras.layers.Dense(1, activation=None)
        # self.final_nccl = tf.keras.layers.Dense(1, activation=None)

    def set_graph(self, graph):
        self.graph = graph

    def set_groups(self, op_groups, device_groups):
        self.op_groups = op_groups
        self.device_groups = device_groups

    def call(self, inputs):
        [op_feats, device_feats, tensor_feats, link_feats, placement_feats, op_types] = inputs

        op_embedding = self.op_embedding(tf.expand_dims(op_types, 1)) # shape: (n_nodes, 1, op_embedding_len)
        op_feats = tf.concat([op_feats, tf.squeeze(op_embedding, axis=1)], 1)

        op_feats = self.op_trans(op_feats)
        device_feats = self.device_trans(device_feats)

        edge_feats = {
            "link": link_feats,
            "prev": tensor_feats,
            "succ": tensor_feats,
            "place": placement_feats,
            "serve": placement_feats
        }
        edge_feats = { etype: self.edge_trans[etype](edge_feats[etype]) for etype in all_etypes }

        for gconv_layer in self.gconv_layers:
            op_feats, device_feats = gconv_layer(self.graph, op_feats, device_feats, edge_feats)

        # if self.cgroups is not None:
        #     c_embedding = tf.concat([tf.expand_dims(tf.math.add_n([c_embedding[i, :] for i in group]), 0) for group in self.cgroups], 0)

        # if self.tgroups is not None:
        #     t_embedding = tf.concat([tf.expand_dims(tf.math.add_n([t_embedding[i, :] for i in group]), 0) for group in self.tgroups], 0)

        return tf.matmul(op_feats, device_feats, transpose_b=True)
