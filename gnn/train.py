import numpy as np
import time
import tensorflow as tf

from data import get_all_data
from model import Model
from environment import evaluate_logp

import sys
def info(*args):
    print(*args, file=sys.stderr, flush=True)

records = get_all_data()

with tf.device("/gpu:0"):
    model = Model(4, 1, 2, 2)

    try:
        # model.load_weights('weights')
        info("load saved weight")
    except:
        info("no saved weight")

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.00001)

    for epoch in range(1000):
        record = records[np.random.randint(len(records))]

        cnfeats = tf.convert_to_tensor(record["cnfeats"], dtype=tf.float32)
        cefeats = tf.convert_to_tensor(record["cefeats"], dtype=tf.float32)
        tnfeats = tf.convert_to_tensor(record["tnfeats"], dtype=tf.float32)
        tefeats = tf.convert_to_tensor(record["tefeats"], dtype=tf.float32)
        model.set_graphs(record["cgraph"], record["tgraph"])

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)
            logp = model([cnfeats, cefeats, tnfeats, tefeats], training=True)
            results = evaluate_logp(record, logp.numpy()) # numpy to turn off gradient tracking
            lps = [tf.reduce_sum(tf.boolean_mask(logp, mask)) for mask, loss_env in results]
            lsp = tf.reduce_logsumexp(tf.concat([tf.expand_dims(x, 0) for x in lps], 0))
            loss = tf.add_n([-loss_env * (lp - lsp) for (_, loss_env), lp in zip(results, lps)])
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if epoch % 10 == 0:
            model.save_weights('weights')

        p = np.argmax(logp.numpy(), axis=2)
        count = {}
        for i in range(p.shape[0]):
            d = tuple(p[i, :])
            count[d] = count.get(d, 0) + 1
        info(count)

        info("loss: ", loss.numpy() / 1000000)
