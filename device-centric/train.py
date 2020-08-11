import numpy as np
import time
import tensorflow as tf

from data import get_all_data
from model import Model
from environment import sample, evaluate_logp, evaluate
from utils import save, load

import sys
def info(*args):
    print(*args, file=sys.stdout, flush=True)

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

    optimizer = tf.keras.optimizers.Adam(learning_rate=.000001, clipnorm=6)
    col_diversity_factor = .5
    L2_regularization_factor = .000001

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
            loss = 0
            nccllogp, logp = model([cnfeats, cefeats, cntypes, tnfeats, tefeats], training=True)
            # info(np.exp(logp.numpy()))
            for _ in range(4):
                ncclmask, mask, loss_rel = evaluate_logp(record, nccllogp.numpy(), logp.numpy()) # numpy to turn off gradient tracking
                loss += loss_rel * (tf.reduce_mean(logp * mask) + tf.reduce_mean(nccllogp * ncclmask))

            if col_diversity_factor > 0: # add diversity for different placements
                negative_col_diversity = tf.reduce_mean(tf.square(tf.reduce_mean(tf.exp(logp[:, 1:]), axis=0)))
                # info(loss.numpy(), col_diversity_factor * negative_col_diversity.numpy())
                loss += col_diversity_factor * negative_col_diversity

            if L2_regularization_factor > 0:
                for weight in model.trainable_weights:
                    loss += L2_regularization_factor * tf.nn.l2_loss(weight)

            grads = tape.gradient(loss, model.trainable_weights)
            # info([tf.reduce_mean(tf.abs(grad)).numpy() for grad in grads])
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if epoch % 10 == 0:
            model.save_weights('weights')
            save(records, "records")

        _ncclmask, _mask = sample(nccllogp), sample(logp)
        p = [ [int(_ncclmask[gi])] + [ int(_mask[j, gi]) for j in range(_mask.shape[0]) ] for gi, group in enumerate(record["cgroups"]) ]
        count = {}
        for l in p:
            d = tuple(l)
            count[d] = count.get(d, 0) + 1
        for d, c in sorted(list(count.items()), key=lambda x: -x[1]):
            info(f"{d}: {c}")
        info("loss: ", loss_rel)
