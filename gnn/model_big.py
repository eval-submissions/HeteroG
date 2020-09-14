import tensorflow as tf
import dgl.function as fn
import numpy as np
from utils import info

class GATConv(layers.Layer):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        xinit = tf.keras.initializers.VarianceScaling(scale=np.sqrt(
            2), mode="fan_avg", distribution="untruncated_normal")
        if isinstance(in_feats, tuple):
            self.fc_src = layers.Dense(
                out_feats * num_heads, use_bias=False, kernel_initializer=xinit)
            self.fc_dst = layers.Dense(
                out_feats * num_heads, use_bias=False, kernel_initializer=xinit)
        else:
            self.fc = layers.Dense(
                out_feats * num_heads, use_bias=False, kernel_initializer=xinit)
        self.attn_l = tf.Variable(initial_value=xinit(
            shape=(1, num_heads, out_feats), dtype='float32'), trainable=True)
        self.attn_r = tf.Variable(initial_value=xinit(
            shape=(1, num_heads, out_feats), dtype='float32'), trainable=True)
        self.feat_drop = layers.Dropout(rate=feat_drop)
        self.attn_drop = layers.Dropout(rate=attn_drop)
        self.leaky_relu = layers.LeakyReLU(alpha=negative_slope)
        if residual:
            if in_feats != out_feats:
                self.res_fc = layers.Dense(
                    num_heads * out_feats, use_bias=False, kernel_initializer=xinit)
            else:
                self.res_fc = Identity()
        else:
            self.res_fc = None
            # self.register_buffer('res_fc', None)
        self.activation = activation

    def call(self, graph, feat):
        with graph.local_scope():
            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    self.fc_src, self.fc_dst = self.fc, self.fc
                feat_src = tf.reshape(self.fc_src(h_src), (-1, self._num_heads, self._out_feats))
                feat_dst = tf.reshape(self.fc_dst(h_dst), (-1, self._num_heads, self._out_feats))
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = tf.reshape(
                    self.fc(h_src), (-1, self._num_heads, self._out_feats))
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = tf.reduce_sum(feat_src * self.attn_l, axis=-1, keepdims=True)
            er = tf.reduce_sum(feat_dst * self.attn_r, axis=-1, keepdims=True)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = tf.reshape(self.res_fc(
                    h_dst), (h_dst.shape[0], -1, self._out_feats))
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst

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
            GConv(node_hidden, node_hidden, edge_hidden, tf.math.tanh),
            GConv(node_hidden, node_hidden, edge_hidden, tf.math.tanh)
        ]

        self.c_corss_layers = [
            Cross(node_hidden),
            Cross(node_hidden),
        ]

        self.t_gconv_layers = [
            GConv(node_hidden, node_hidden, edge_hidden, tf.math.tanh),
            GConv(node_hidden, node_hidden, edge_hidden, tf.math.tanh)
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
