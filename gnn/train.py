import numpy as np
import time
import tensorflow as tf

from data import get_all_data
from model import Model
from environment import sample, evaluate, sample_and_evaluate
from utils import save, load, info

try:
    records = load("records")
    info("load saved records")
except:
    records = get_all_data()
    info("no saved records")
    save(records, "records")

with tf.device("/gpu:0"):
    model = Model(records[0]["op_table"])

    try:
        model.load_weights('weights')
        info("load saved weight")
        init = 1000
    except:
        info("no saved weight")
        init = 0

    L2_regularization_factor = .0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=.002, clipnorm=6)

    for epoch in range(init, 20000):
        if epoch == 100:
            optimizer = tf.keras.optimizers.Adam(learning_rate=.01, clipnorm=6)
        elif epoch == 1000:
            optimizer = tf.keras.optimizers.Adam(learning_rate=.001, clipnorm=1)

        record_id = np.random.randint(len(records))
        record = records[record_id]
        # record = records[1]

        op_types = tf.convert_to_tensor(record["op_types"], dtype=tf.float32)
        op_feats = tf.convert_to_tensor(record["op_feats"], dtype=tf.float32)
        device_feats = tf.convert_to_tensor(record["device_feats"], dtype=tf.float32)
        tensor_feats = tf.convert_to_tensor(record["tensor_feats"], dtype=tf.float32)
        link_feats = tf.convert_to_tensor(record["link_feats"], dtype=tf.float32)
        model.set_graph(record["graph"])

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)

            strategy = np.zeros((op_feats.shape[0], device_feats.shape[0]), dtype=np.float32)
            negative_logp = 0
            for gid in range(len(record['op_groups'])):
                placement_feats = [ [strategy[i,j]] for i in range(strategy.shape[0]) for j in range(strategy.shape[1]) ]
                placement_feats = tf.convert_to_tensor(placement_feats, dtype=tf.float32)

                nodes_to_place = [ 1 if i in record['op_groups'][gid] else 0 for i in range(op_feats.shape[0]) ]
                nodes_to_place = tf.convert_to_tensor(nodes_to_place, dtype=tf.float32)
                nodelogit = model([op_feats, device_feats, tensor_feats, link_feats, placement_feats, op_types, nodes_to_place], training=True) # TODO: nodes that are already placed
                nodelogit = tf.reshape(nodelogit, strategy.shape)

                group_mask = np.zeros(nodelogit.shape, dtype=np.float32)
                for i in record['op_groups'][gid]:
                    group_mask[i, :] = 1
                p = sample(nodelogit.numpy())
                negative_logp += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(tf.convert_to_tensor(p, dtype=tf.float32), nodelogit) * group_mask)
                for i in record['op_groups'][gid]:
                    strategy[i, :] = p[i, :]

            sqrt_time = evaluate(record, np.zeros((op_feats.shape[0],)), strategy)[0]
            if "hist" not in record:
                record["hist"] = [sqrt_time]
            else:
                record["hist"].append(sqrt_time)
                record["hist"] = record["hist"][-100:]
            baseline = np.mean(record["hist"])
            advantage = -(sqrt_time - baseline) / baseline
            loss = advantage * negative_logp

                # ncclmask, nodemask, advantage, sqrt_time, oom, leftout = sample_and_evaluate(record, nccllogit.numpy(), nodelogit.numpy()) # numpy to turn off gradient tracking
                # for i in oom:
                #     negative_logp = tf.nn.sigmoid_cross_entropy_with_logits([0.] * nodelogit.shape[1], nodelogit[i, :])
                #     loss += .01 * tf.reduce_sum(negative_logp)
                # for gi in leftout:
                #     negative_logp = tf.nn.sigmoid_cross_entropy_with_logits([1.] * nodelogit.shape[0], nodelogit[:, gi])
                #     loss += .01 * tf.reduce_sum(negative_logp)
                # if len(oom) == 0: # and len(leftout) == 0:
                #     negative_ncclunionlogp = tf.nn.sigmoid_cross_entropy_with_logits(ncclmask.astype(np.float32), nccllogit)
                #     negative_nodeunionlogp = tf.nn.sigmoid_cross_entropy_with_logits(nodemask.astype(np.float32), nodelogit)
                #     loss += advantage * (tf.reduce_sum(negative_ncclunionlogp) + tf.reduce_sum(negative_nodeunionlogp))
                #     info(advantage)

                # loss += (time_predict[0, 0] - time) * (time_predict[0, 0] - time)
                # loss += tf.reduce_sum((memory_predict[:, 0] - memory) * (memory_predict[:, 0] - memory))

                # info(loss)

            if epoch % 10 == 0:
                info(record_id, sqrt_time, evaluate(record, np.zeros((op_feats.shape[0],)), np.ones(strategy.shape))[0])

            if L2_regularization_factor > 0:
                for weight in model.trainable_weights:
                    loss += L2_regularization_factor * tf.nn.l2_loss(weight)

            grads = tape.gradient(loss, model.trainable_weights)
            # info([tf.reduce_mean(tf.abs(grad)).numpy() for grad in grads])
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if epoch % 50 == 0:
            info("==== save ====")
            model.save_weights('weights')
            save(records, "records")

        # ncclmask, nodemask, advantage, sqrt_time, oom, leftout = sample_and_evaluate(record, nccllogit.numpy(), nodelogit.numpy())
        # p = [ [int(ncclmask[gi])] + [ int(nodemask[j, gi]) for j in range(nodemask.shape[0]) ] for gi, group in enumerate(record["cgroups"]) ]
        # count = {}
        # for l in p:
        #     d = tuple(l)
        #     count[d] = count.get(d, 0) + 1
        # for d, c in sorted(list(count.items()), key=lambda x: -x[1]):
        #     info(f"{d}: {c}")
        # info("time: ", sqrt_time, oom, leftout)

        # p = sample(nodelogit)
        # info(strategy.astype(np.int))
        # info(p)
