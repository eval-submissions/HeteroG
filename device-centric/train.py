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
    model = Model(4, 2, 2, 3, 20, records[0]["op_table"])

    try:
        model.load_weights('weights')
        info("load saved weight")
    except:
        info("no saved weight")

    optimizer = tf.keras.optimizers.Adam(learning_rate=.001, clipnorm=1)
    L2_regularization_factor = .0005

    for epoch in range(20000):
        record = records[np.random.randint(len(records))]

        cnfeats = tf.convert_to_tensor(record["cnfeats"], dtype=tf.float32)
        cefeats = tf.convert_to_tensor(record["cefeats"], dtype=tf.float32)
        cntypes = tf.convert_to_tensor(record["cntypes"], dtype=tf.float32)
        tnfeats = tf.convert_to_tensor(record["tnfeats"], dtype=tf.float32)
        tefeats = tf.convert_to_tensor(record["tefeats"], dtype=tf.float32)
        model.set_graphs(record["cgraph"], record["tgraph"])
        model.set_groups(record["cgroups"], record["tgroups"])

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)
            nccllogit, nodelogit = model([cnfeats, cefeats, cntypes, tnfeats, tefeats], training=True)
            loss = 0
            for _ in range(4):
                ncclmask, nodemask, advantage, sqrt_time, oom, leftout = sample_and_evaluate(record, nccllogit.numpy(), nodelogit.numpy()) # numpy to turn off gradient tracking
                for i in oom:
                    negative_logp = tf.nn.sigmoid_cross_entropy_with_logits([0.] * nodelogit.shape[1], nodelogit[i, :])
                    loss += .005 * tf.reduce_sum(negative_logp)
                for gi in leftout:
                    negative_logp = tf.nn.sigmoid_cross_entropy_with_logits([1.] * nodelogit.shape[0], nodelogit[:, gi])
                    loss += .002 * tf.reduce_sum(negative_logp)
                negative_ncclunionlogp = tf.nn.sigmoid_cross_entropy_with_logits(ncclmask.astype(np.float32), nccllogit)
                negative_nodeunionlogp = tf.nn.sigmoid_cross_entropy_with_logits(nodemask.astype(np.float32), nodelogit)
                loss += -advantage * -(tf.reduce_sum(negative_ncclunionlogp) + tf.reduce_sum(negative_nodeunionlogp))
                info(advantage)

            if L2_regularization_factor > 0:
                for weight in model.trainable_weights:
                    loss += L2_regularization_factor * tf.nn.l2_loss(weight)

            grads = tape.gradient(loss, model.trainable_weights)
            info([tf.reduce_mean(tf.abs(grad)).numpy() for grad in grads])
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if epoch % 10 == 0:
            model.save_weights('weights')
            save(records, "records")

        ncclmask, nodemask, advantage, sqrt_time, oom, leftout = sample_and_evaluate(record, nccllogit.numpy(), nodelogit.numpy())
        p = [ [int(ncclmask[gi])] + [ int(nodemask[j, gi]) for j in range(nodemask.shape[0]) ] for gi, group in enumerate(record["cgroups"]) ]
        count = {}
        for l in p:
            d = tuple(l)
            count[d] = count.get(d, 0) + 1
        for d, c in sorted(list(count.items()), key=lambda x: -x[1]):
            info(f"{d}: {c}")
        info("time: ", sqrt_time, oom, leftout)
