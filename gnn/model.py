import tensorflow as tf
import dgl.function as fn
import numpy as np

import sys
def info(*args):
    print(*args, file=sys.stdout, flush=True)

class GConv(tf.keras.layers.Layer):
    '''Graph Conv layer that concats the edge features before sending message'''
    def __init__(self, in_feats, out_feats, edge_feats, residual=False, activation=None):
        super(GConv, self).__init__()
        self.in_feats = in_feats + edge_feats
        self.out_feats = out_feats

        xinit = tf.keras.initializers.glorot_uniform()
        self.weight = tf.Variable(initial_value=xinit(
            shape=(self.in_feats, self.out_feats), dtype='float32'), trainable=True)

        zeroinit = tf.keras.initializers.zeros()
        self.bias = tf.Variable(initial_value=zeroinit(
            shape=(self.out_feats, ), dtype='float32'), trainable=True)

        self.activation = activation
        self.residual = residual

    def call(self, graph, feat, edge_feat):
        graph = graph.local_var()

        degs = tf.clip_by_value(tf.cast(graph.out_degrees(), tf.float32),
                                clip_value_min=1, clip_value_max=np.inf)
        norm = tf.pow(degs, -0.5)
        shp = norm.shape + (1,) * (feat.ndim - 1)
        norm = tf.reshape(norm, shp)
        feat = feat * norm
        # TODO: normalize edge feature

        graph.srcdata['h'] = feat
        graph.edata['e'] = edge_feat
        graph.update_all(lambda edge: {'m': tf.concat([edge.src['h'], edge.data['e']], axis=1)}, fn.sum(msg='m', out='h'))
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

        if self.residual:
            rst = feat + rst

        return rst

class Model(tf.keras.Model):
    def __init__(self, cfeat_len, cedge_len, tfeat_len, tedge_len, op_table):
        super(Model, self).__init__()

        num_hidden = 256
        num_rnn_hidden = 128
        op_embedding_len = 4

        self.op_embedding = tf.keras.layers.Embedding(len(op_table), op_embedding_len, input_length=1)

        self.c_gconv_layers = [
            GConv(cfeat_len + op_embedding_len, num_hidden, cedge_len, False, tf.sigmoid),
            GConv(num_hidden, num_hidden, cedge_len, True, tf.sigmoid),
            GConv(num_hidden, num_hidden, cedge_len, True, tf.sigmoid),
            GConv(num_hidden, num_hidden, cedge_len, True, tf.sigmoid),
            GConv(num_hidden, num_hidden, cedge_len, True, tf.sigmoid),
            GConv(num_hidden, num_hidden, cedge_len, False, None)
        ]

        self.t_gconv_layers = [
            GConv(tfeat_len, num_hidden, tedge_len, False, tf.sigmoid),
            GConv(num_hidden, num_hidden, tedge_len, True, tf.sigmoid),
            GConv(num_hidden, num_hidden, tedge_len, True, tf.sigmoid),
            GConv(num_hidden, num_hidden, tedge_len, False, None)
        ]

        self.rnn_layers = [
            # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_rnn_hidden, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_rnn_hidden, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_rnn_hidden, return_sequences=True))
        ]

        self.final = tf.keras.layers.Dense(2, activation=tf.nn.log_softmax) # put 0 or 1 replicas

    def set_graphs(self, cgraph, tgraph):
        self.cgraph = cgraph
        self.tgraph = tgraph

    def set_groups(self, groups):
        self.groups = groups

    def call(self, inputs):
        [cfeats, cedge_feats, ctypes, tfeats, tedge_feats, combined_feats] = inputs

        op_embedding = self.op_embedding(tf.expand_dims(ctypes, 1)) # shape: (n_nodes, 1, op_embedding_len)
        x = tf.concat([cfeats, tf.squeeze(op_embedding, axis=1)], 1)
        for layer in self.c_gconv_layers:
            x = layer(self.cgraph, x, cedge_feats)
            x = tf.reshape(x, (x.shape[0], -1))
        c_embedding = x

        x = tfeats
        for layer in self.t_gconv_layers:
            x = layer(self.tgraph, x, tedge_feats)
            x = tf.reshape(x, (x.shape[0], -1))
        t_embedding = x

        if self.groups is not None:
            c_embedding = tf.concat([tf.expand_dims(tf.math.add_n([c_embedding[i, :] for i in group]) / len(group), 0) for group in self.groups], 0)

        batches = []
        for i in range(c_embedding.shape[0]):
            x = tf.repeat(tf.reshape(c_embedding[i, :], (1, c_embedding.shape[1])), repeats=[t_embedding.shape[0]], axis=0)
            x = tf.concat([x, t_embedding], 1) # TODO: add combination features (running time of a node in a device) here
            batches.append(tf.expand_dims(x, 0))
        x = tf.concat(batches, 0) # [batchsize, seq_len, num_feature]
        for layer in self.rnn_layers:
            x = layer(x)
        x = self.final(x) # the Dense layer is applied on the last axis of input tensor (https://github.com/tensorflow/tensorflow/issues/30882)

        return x
