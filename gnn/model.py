import tensorflow as tf
import dgl.function as fn
import numpy as np
from utils import info

class GConv(tf.keras.layers.Layer):
    '''Graph Conv layer that concats the edge features before sending message'''
    def __init__(self, out_feats, activation=None):
        super(GConv, self).__init__()
        self.dense = tf.keras.layers.Dense(out_feats, activation=activation)

    def call(self, graph, feat, edge_feat):
        graph = graph.local_var()

        graph.srcdata['h'] = feat
        graph.edata['e'] = edge_feat
        # TODO: use built-in method to concat src to edge and run dense layer on graph.edata may be faster
        def update(edge):
            m = tf.concat([edge.src['h'], edge.data['e']], axis=1)
            return {'m': m}
        graph.update_all(update, fn.sum(msg='m', out='h'))
        rst = self.dense(graph.dstdata['h'])

        return feat + rst

class Cross(tf.keras.layers.Layer):
    '''Attention layer across two graphs'''
    def __init__(self, feature_len):
        super(Cross, self).__init__()
        self.query_transform1 = tf.keras.layers.Dense(feature_len // 4, activation=None)
        self.key_transform1 = tf.keras.layers.Dense(feature_len // 4, activation=None)
        self.value_transform1 = tf.keras.layers.Dense(feature_len, activation=None)

        self.query_transform2 = tf.keras.layers.Dense(feature_len // 4, activation=None)
        self.key_transform2 = tf.keras.layers.Dense(feature_len // 4, activation=None)
        self.value_transform2 = tf.keras.layers.Dense(feature_len, activation=None)

        self.dense = tf.keras.layers.Dense(feature_len, activation=tf.math.tanh)

    def call(self, this, that, mask):
        s1 = tf.matmul(self.query_transform1(this), self.key_transform1(that), transpose_b=True)
        w1 = tf.nn.softmax(s1)
        r1 = tf.matmul(w1, self.value_transform1(that))

        s2 = tf.matmul(self.query_transform2(this), self.key_transform2(that), transpose_b=True)
        s2 -= 1.e9 * tf.cast(tf.logical_not(mask), dtype=tf.float32)
        w2 = tf.nn.softmax(s2)
        r2 = tf.matmul(w2, self.value_transform2(that))

        return self.dense(tf.concat([r1, r2], 1)) + this

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

        node_hidden = 256
        edge_hidden = 8
        op_embedding_len = 8
        c_embedding_len = 1024
        t_embedding_len = 1024

        self.op_embedding = tf.keras.layers.Embedding(len(op_table), op_embedding_len, input_length=1)

        self.cntrans = tf.keras.layers.Dense(node_hidden, activation=tf.math.tanh)
        self.cetrans = tf.keras.layers.Dense(edge_hidden, activation=tf.math.tanh)
        self.tntrans = tf.keras.layers.Dense(node_hidden, activation=tf.math.tanh)
        self.tetrans = tf.keras.layers.Dense(edge_hidden, activation=tf.math.tanh)

        self.c_gconv_layers = [
            GConv(node_hidden, tf.math.tanh),
            GConv(node_hidden, tf.math.tanh)
        ]

        self.c_corss_layers = [
            Cross(node_hidden),
            Cross(node_hidden),
        ]

        self.t_gconv_layers = [
            GConv(node_hidden, tf.math.tanh),
            GConv(node_hidden, tf.math.tanh)
        ]

        self.t_corss_layers = [
            Cross(node_hidden),
            Cross(node_hidden),
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
