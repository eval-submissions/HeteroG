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
        def update(edge):
            m = tf.concat([edge.src['h'], edge.data['e']], axis=1)
            if self.activation is not None:
                m = self.activation(m)
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

        self.final = tf.keras.layers.Dense(6, activation=tf.nn.log_softmax)

    def set_graphs(self, cgraph, tgraph):
        self.cgraph = cgraph
        self.tgraph = tgraph

    def set_groups(self, cgroups, tgroups):
        self.cgroups = cgroups
        self.tgroups = tgroups

    def call(self, inputs):
        [cfeats, cedge_feats, ctypes, tfeats, tedge_feats] = inputs

        op_embedding = self.op_embedding(tf.expand_dims(ctypes, 1)) # shape: (n_nodes, 1, op_embedding_len)
        x = tf.concat([cfeats, tf.squeeze(op_embedding, axis=1)], 1)
        for layer in self.c_gconv_layers:
            x = layer(self.cgraph, x, cedge_feats)
            x = tf.reshape(x, (x.shape[0], -1))
        c_embedding = x

        if self.cgroups is not None:
            c_embedding = tf.concat([tf.expand_dims(tf.math.add_n([c_embedding[i, :] for i in group]) / len(group), 0) for group in self.cgroups], 0)

        x = tfeats
        for layer in self.t_gconv_layers:
            x = layer(self.tgraph, x, tedge_feats)
            x = tf.reshape(x, (x.shape[0], -1))
        t_embedding = x

        if self.tgroups is not None:
            t_embedding = tf.concat([tf.expand_dims(tf.math.add_n([t_embedding[i, :] for i in group]), 0) for group in self.tgroups], 0)

        x = tf.repeat(tf.reshape(t_embedding, (1, -1)), repeats=[c_embedding.shape[0]], axis=0)
        x = tf.concat([c_embedding, x], 1) # [n_node, c_embedding_len + 6 * t_embedding_len]
        x = self.final(x) # [n_node, 6]

        return x
