import tensorflow as tf
from tensorflow.keras import layers
import dgl.function as fn
from dgl.nn.tensorflow import edge_softmax, GATConv

class GAT(tf.keras.Model):
    def __init__(self, computation_feature_length, device_feature_length):
        super(GAT, self).__init__()

        num_hidden = 8
        num_heads = 8
        GAT_options = (0.5, 0.5, 0.2) # feat_drop_rate, attn_drop_rate, negative_slope
        num_rnn_hidden = 8

        self.computation_gat_layers = [
            GATConv(computation_feature_length, num_hidden, num_hidden, *GAT_options, False, tf.nn.elu),
            GATConv(num_hidden * num_heads, num_hidden, num_heads, *GAT_options, False, tf.nn.elu),
            GATConv(num_hidden * num_heads, num_hidden, num_heads, *GAT_options, False, tf.nn.elu),
            GATConv(num_hidden * num_heads, num_hidden, 1, *GAT_options, False, None)
        ]

        self.device_gat_layers = [
            GATConv(device_feature_length, num_hidden, num_heads, *GAT_options, False, tf.nn.elu),
            GATConv(num_hidden * num_heads, num_hidden, num_heads, *GAT_options, False, tf.nn.elu),
            GATConv(num_hidden * num_heads, num_hidden, num_heads, *GAT_options, False, tf.nn.elu),
            GATConv(num_hidden * num_heads, num_hidden, 1, *GAT_options, False, None)
        ]

        self.rnn_layers = [
            tf.keras.layers.Bidirectional(tf.keras.layers.GRU(num_rnn_hidden, return_sequences=True))
        ]

        self.final = tf.keras.layers.Dense(3, activation='softmax') # put 0, 1, 2 replicas

    def set_graphs(self, computation_graph, device_graph):
        self.computation_graph = computation_graph
        self.device_graph = device_graph

    def call(self, inputs):
        [computation_features, device_features] = inputs

        x = computation_features
        for layer in self.computation_gat_layers:
            x = layer(self.computation_graph, x)
            x = tf.reshape(x, (x.shape[0], -1))
        computation_embedding = x

        x = device_features
        for layer in self.device_gat_layers:
            x = layer(self.device_graph, x)
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
        x = self.final(x) # contrary to what has been stated in documentation, the Dense layer is applied on the last axis of input tensor (https://github.com/tensorflow/tensorflow/issues/30882)

        return x
