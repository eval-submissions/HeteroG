import tensorflow as tf
import dgl.function as fn
import numpy as np
from utils import info

class GConv(tf.keras.layers.Layer):
    '''Graph Conv layer that concats the edge features before sending message'''
    def __init__(self, out_feats, activation=None):
        super(GConv, self).__init__()
        self.dense = tf.keras.layers.Dense(out_feats, activation=activation)
        self.dense2 = tf.keras.layers.Dense(out_feats, activation=None)

    def call(self, graph, feat, edge_feat):
        graph = graph.local_var()

        graph.srcdata['h'] = feat
        graph.apply_edges(fn.copy_u('h', 's'))
        graph.edata['e'] = self.dense(tf.concat([graph.edata.pop('s'), edge_feat], axis=1))
        graph.update_all(fn.copy_e('e', 'm'), fn.sum(msg='m', out='h'))
        rst = self.dense2(graph.dstdata['h'])

        return feat + rst

class Cross(tf.keras.layers.Layer):
    '''Attention layer across two graphs'''
    def __init__(self, feature_len, activation=None):
        super(Cross, self).__init__()
        self.query_transform = [ tf.keras.layers.Dense(feature_len // 4, activation=None) for _ in range(8) ]
        self.key_transform = [ tf.keras.layers.Dense(feature_len // 4, activation=None) for _ in range(8) ]
        self.value_transform = [ tf.keras.layers.Dense(feature_len // 4, activation=None) for _ in range(8) ]

        self.dense = tf.keras.layers.Dense(feature_len, activation=activation)

    def call(self, this, that, mask):
        rs = []
        for i in range(8):
            s = tf.matmul(self.query_transform[i](this), self.key_transform[i](that), transpose_b=True)
            if i >= 4:
                s -= 1.e9 * tf.cast(tf.logical_not(mask), dtype=tf.float32)
            w = tf.nn.softmax(s)
            r = tf.matmul(w, self.value_transform[i](that))
            rs.append(r)

        rst = self.dense(tf.concat(rs, 1))

        return rst + this

# class Cross(tf.keras.layers.Layer):
#     def __init__(self):
#         super(Cross, self).__init__()
#         self.dense = tf.keras.layers.Dense(512, activation=tf.math.tanh)

#     def call(self, this, that, mask):
#         those = tf.matmul(tf.cast(mask, tf.float32), that)

#         return self.dense(tf.concat([this, those], 1)) + this

class Model(tf.keras.Model):
    def __init__(self, op_table):
        super(Model, self).__init__()

        node_hidden = 128
        edge_hidden = 8
        op_embedding_len = 8
        c_embedding_len = 256
        t_embedding_len = 256

        self.op_embedding = tf.keras.layers.Embedding(len(op_table), op_embedding_len, input_length=1)

        self.cntrans = tf.keras.layers.Dense(node_hidden, activation=tf.nn.relu)
        self.cetrans = tf.keras.layers.Dense(edge_hidden, activation=tf.nn.relu)
        self.tntrans = tf.keras.layers.Dense(node_hidden, activation=tf.nn.relu)
        self.tetrans = tf.keras.layers.Dense(edge_hidden, activation=tf.nn.relu)

        self.c_gconv_layers = [
            GConv(node_hidden, tf.nn.relu),
            GConv(node_hidden, tf.nn.relu)
        ]

        self.c_corss_layers = [
            Cross(node_hidden, tf.nn.relu),
            Cross(node_hidden, tf.nn.relu),
        ]

        self.t_gconv_layers = [
            GConv(node_hidden, tf.nn.relu),
            GConv(node_hidden, tf.nn.relu)
        ]

        self.t_corss_layers = [
            Cross(node_hidden, tf.nn.relu),
            Cross(node_hidden, tf.nn.relu),
        ]

        self.c_final = tf.keras.layers.Dense(c_embedding_len, activation=None)
        self.t_final = tf.keras.layers.Dense(t_embedding_len, activation=None)

        # self.final_strategy = tf.keras.layers.Dense(cgroups_len, activation=None)
        # self.final_nccl = tf.keras.layers.Dense(1, activation=None)

    def set_graphs(self, cgraph, tgraph):
        self.cgraph = cgraph
        self.tgraph = tgraph

    def set_groups(self, cgroups, tgroups):
        self.cgroups = cgroups
        self.tgroups = tgroups

    def call(self, inputs):
        [cfeats, cedge_feats, ctypes, tfeats, tedge_feats, placement, runtime_stats] = inputs

        op_embedding = self.op_embedding(tf.expand_dims(ctypes, 1)) # shape: (n_nodes, 1, op_embedding_len)
        cfeats = tf.concat([cfeats, tf.squeeze(op_embedding, axis=1)], 1)

        c_embedding = self.cntrans(cfeats)
        cedge_feats = self.cetrans(cedge_feats)
        t_embedding = self.tntrans(tfeats)
        tedge_feats = self.tetrans(tedge_feats)

        for i in range(len(self.c_gconv_layers)):
            c_embedding = self.c_gconv_layers[i](self.cgraph, c_embedding, cedge_feats)
            t_embedding = self.t_gconv_layers[i](self.tgraph, t_embedding, tedge_feats)

            c_embedding = self.c_corss_layers[i](c_embedding, t_embedding, placement)
            t_embedding = self.t_corss_layers[i](t_embedding, c_embedding, tf.transpose(placement))

        c_embedding = self.c_final(c_embedding)
        t_embedding = self.t_final(t_embedding)

        if self.cgroups is not None:
            c_embedding = tf.concat([tf.expand_dims(tf.math.add_n([c_embedding[i, :] for i in group]), 0) for group in self.cgroups], 0)

        if self.tgroups is not None:
            t_embedding = tf.concat([tf.expand_dims(tf.math.add_n([t_embedding[i, :] for i in group]), 0) for group in self.tgroups], 0)

        return tf.matmul(c_embedding, t_embedding, transpose_b=True)
