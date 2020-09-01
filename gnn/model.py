import tensorflow as tf
import dgl.function as fn
import numpy as np
from utils import info

class GConv(tf.keras.layers.Layer):
    '''Graph Conv layer that concats the edge features before sending message'''
    def __init__(self, in_feats, out_feats, edge_feats, activation=None):
        super(GConv, self).__init__()
        self.in_feats = in_feats + edge_feats
        self.out_feats = out_feats

        xinit = tf.keras.initializers.glorot_uniform()
        self.weight = tf.Variable(initial_value=xinit(
            shape=(self.in_feats, self.out_feats), dtype='float32'), trainable=True)

        zeroinit = tf.keras.initializers.zeros()
        self.bias = tf.Variable(initial_value=zeroinit(
            shape=(self.out_feats, ), dtype='float32'), trainable=True)

        # self.dense = tf.keras.layers.Dense(out_feats, activation=activation)

        self.activation = activation

    def call(self, graph, feat, edge_feat):
        graph = graph.local_var()

        degs = tf.clip_by_value(tf.cast(graph.out_degrees(), tf.float32),
                                clip_value_min=1, clip_value_max=np.inf)
        norm = tf.pow(degs, -0.5)
        shp = norm.shape + (1,) * (feat.ndim - 1)
        norm = tf.reshape(norm, shp)
        feat = feat * norm
        # TODO: normalize edge feature?

        graph.srcdata['h'] = feat
        graph.edata['e'] = edge_feat
        def update(edge):
            m = tf.concat([edge.src['h'], edge.data['e']], axis=1)
            return {'m': m}
        graph.update_all(update, fn.sum(msg='m', out='h'))
        rst = graph.dstdata['h']
        rst = tf.matmul(rst, self.weight)

        degs = tf.clip_by_value(tf.cast(graph.in_degrees(), tf.float32),
                                clip_value_min=1, clip_value_max=np.inf)
        norm = tf.pow(degs, -0.5)
        shp = norm.shape + (1,) * (feat.ndim - 1)
        norm = tf.reshape(norm, shp)
        rst = rst * norm + self.bias

        if self.activation is not None:
            rst = self.activation(rst)

        # rst = self.dense(tf.concat([feat, rst], axis=1))

        return rst

class Model(tf.keras.Model):
    def __init__(self, cfeat_len, cedge_len, tfeat_len, tedge_len, cgroups_len, op_table):
        super(Model, self).__init__()

        num_hidden = 512
        op_embedding_len = 8
        c_embedding_len = 64
        t_embedding_len = 64

        self.op_embedding = tf.keras.layers.Embedding(len(op_table), op_embedding_len, input_length=1)

        self.c_gconv_layers = [
            GConv(cfeat_len + op_embedding_len, num_hidden, cedge_len, tf.nn.relu),
            GConv(num_hidden, num_hidden, cedge_len, tf.nn.relu),
            GConv(num_hidden, num_hidden, cedge_len, tf.nn.relu),
            GConv(num_hidden, num_hidden, cedge_len, tf.nn.relu),
            GConv(num_hidden, c_embedding_len, cedge_len, None)
        ]

        self.t_gconv_layers = [
            GConv(tfeat_len, num_hidden, tedge_len, tf.nn.relu),
            GConv(num_hidden, num_hidden, tedge_len, tf.nn.relu),
            GConv(num_hidden, num_hidden, tedge_len, tf.nn.relu),
            GConv(num_hidden, num_hidden, tedge_len, tf.nn.relu),
            GConv(num_hidden, t_embedding_len, tedge_len, None)
        ]

        self.final_strategy = tf.keras.layers.Dense(cgroups_len, activation=None)
        self.final_nccl = tf.keras.layers.Dense(1, activation=None)

    def set_graphs(self, cgraph, tgraph):
        self.cgraph = cgraph
        self.tgraph = tgraph

    def set_groups(self, cgroups, tgroups):
        self.cgroups = cgroups
        self.tgroups = tgroups

    def call(self, inputs):
        [cfeats, cedge_feats, ctypes, tfeats, tedge_feats] = inputs

        op_embedding = self.op_embedding(tf.expand_dims(ctypes, 1)) # shape: (n_nodes, 1, op_embedding_len)
        c_embedding = tf.concat([cfeats, tf.squeeze(op_embedding, axis=1)], 1)
        t_embedding = tfeats

        n_layers = len(self.c_gconv_layers)
        for i in range(n_layers):
            c_embedding = self.c_gconv_layers[i](self.cgraph, c_embedding, cedge_feats)
            t_embedding = self.t_gconv_layers[i](self.tgraph, t_embedding, tedge_feats)

            # if i >= 2:
            #     c_embedding = tf.expand_dims(c_embedding, 0)
            #     t_embedding = tf.expand_dims(t_embedding, 0)

            #     c_embedding = tf.keras.layers.Attention()([c_embedding, t_embedding])
            #     t_embedding = tf.keras.layers.Attention()([t_embedding, c_embedding])

            #     c_embedding = tf.squeeze(c_embedding, axis=0)
            #     t_embedding = tf.squeeze(t_embedding, axis=0)

        if self.cgroups is not None:
            c_embedding = tf.concat([tf.expand_dims(tf.math.add_n([c_embedding[i, :] for i in group]), 0) for group in self.cgroups], 0)

        if self.tgroups is not None:
            t_embedding = tf.concat([tf.expand_dims(tf.math.add_n([t_embedding[i, :] for i in group]), 0) for group in self.tgroups], 0)

        return tf.squeeze(self.final_nccl(c_embedding), axis=1), self.final_strategy(t_embedding)
