import tensorflow as tf
import dgl.function as fn
import numpy as np

class GConv(tf.keras.layers.Layer):
    '''Graph Conv layer that concats the edge features before sending message'''
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation=None):
        super(GConv, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats

        xinit = tf.keras.initializers.glorot_uniform()
        self.weight = tf.Variable(initial_value=xinit(
            shape=(in_feats, out_feats), dtype='float32'), trainable=True)

        zeroinit = tf.keras.initializers.zeros()
        self.bias = tf.Variable(initial_value=zeroinit(
            shape=(out_feats), dtype='float32'), trainable=True)

        self._activation = activation

    def call(self, graph, feat, edge_feat, weight=None):
        graph = graph.local_var()

        degs = tf.clip_by_value(tf.cast(graph.out_degrees(), tf.float32),
                                clip_value_min=1,
                                clip_value_max=np.inf)
        norm = tf.pow(degs, -0.5)
        shp = norm.shape + (1,) * (feat.ndim - 1)
        norm = tf.reshape(norm, shp)
        feat = feat * norm

        if weight is not None:
            raise DGLError('External weight is provided while at the same time the'
                            ' module has defined its own weight parameter. Please'
                            ' create the module with flag weight=False.')
        else:
            weight = self.weight

        graph.srcdata['h'] = feat
        graph.edata['e'] = edge_feat
        graph.update_all(lambda edge: {'m': tf.concat([edge.src['h'], edge.data['e']], axis=1)}, fn.sum(msg='m', out='h'))
        rst = graph.dstdata['h']
        rst = tf.matmul(rst, weight)

        degs = tf.clip_by_value(tf.cast(graph.in_degrees(), tf.float32),
                                clip_value_min=1,
                                clip_value_max=np.inf)
        norm = tf.pow(degs, -0.5)
        shp = norm.shape + (1,) * (feat.ndim - 1)
        norm = tf.reshape(norm, shp)
        rst = rst * norm + self.bias

        if self._activation is not None:
            rst = self._activation(rst)

        return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = 'in={_in_feats}, out={_out_feats}'
        summary += ', normalization={_norm}'
        if '_activation' in self.__dict__:
            summary += ', activation={_activation}'
        return summary.format(**self.__dict__)

class Model(tf.keras.Model):
    def __init__(self, computation_feature_length, device_feature_length):
        super(Model, self).__init__()

        num_hidden = 1024
        num_rnn_hidden = 256

        self.computation_gconv_layers = [
            GConv(computation_feature_length, num_hidden, tf.nn.elu),
            # GConv(num_hidden, num_hidden, tf.nn.elu),
            # GConv(num_hidden, num_hidden, tf.nn.elu),
            # GConv(num_hidden, num_hidden, tf.nn.elu),
            # GConv(num_hidden, num_hidden, tf.nn.elu),
            GConv(num_hidden+1, num_hidden, None)
        ]

        self.device_gconv_layers = [
            GConv(device_feature_length, num_hidden, tf.nn.elu),
            # GConv(num_hidden, num_hidden, tf.nn.elu),
            # GConv(num_hidden, num_hidden, tf.nn.elu),
            # GConv(num_hidden, num_hidden, tf.nn.elu),
            GConv(num_hidden+1, num_hidden, None)
        ]

        self.rnn_layers = [
            # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_rnn_hidden, return_sequences=True)),
            # tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_rnn_hidden, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_rnn_hidden, return_sequences=True))
        ]

        self.final = tf.keras.layers.Dense(3, activation=tf.nn.log_softmax) # put 0, 1, 2 replicas

    def set_graphs(self, computation_graph, device_graph):
        self.computation_graph = computation_graph
        self.device_graph = device_graph

    def call(self, inputs):
        [computation_features, device_features] = inputs

        x = computation_features
        for layer in self.computation_gconv_layers:
            x = layer(self.computation_graph, x, np.array([[1]] * self.computation_graph.number_of_edges(), dtype='float32'))
            x = tf.reshape(x, (x.shape[0], -1))
        computation_embedding = x

        x = device_features
        for layer in self.device_gconv_layers:
            x = layer(self.device_graph, x, np.array([[1]] * self.device_graph.number_of_edges(), dtype='float32'))
            x = tf.reshape(x, (x.shape[0], -1))
        device_embedding = x

        batches = []
        for i in range(computation_embedding.shape[0]):
            x = tf.repeat(tf.reshape(computation_embedding[i, :], (1, computation_embedding.shape[1])), repeats=[device_embedding.shape[0]], axis=0)
            x = tf.concat([x, device_embedding], 1) # TODO: add combination features (running time of a node in a device) here
            batches.append(tf.expand_dims(x, 0))
        x = tf.concat(batches, 0) # [batchsize, seq_len, num_feature]
        for layer in self.rnn_layers:
            x = layer(x)
        x = self.final(x) # the Dense layer is applied on the last axis of input tensor (https://github.com/tensorflow/tensorflow/issues/30882)

        return x
