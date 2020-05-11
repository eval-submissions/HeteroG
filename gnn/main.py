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
    model = Model(4, 2)

    try:
        model.load_weights('weights')
        info("load saved weight")
    except:
        info("no saved weight")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, epsilon=1e-8)

    for epoch in range(1000):
        record = records[np.random.randint(len(records))]

        computation_features = tf.convert_to_tensor(record["computation_features"], dtype=tf.float32)
        device_features = tf.convert_to_tensor(record["device_features"], dtype=tf.float32)
        model.set_graphs(record["computation_graph"], record["device_graph"])

        with tf.GradientTape() as tape:
            tape.watch(model.trainable_weights)
            logp = model([computation_features, device_features], training=True)
            results = evaluate_logp(record, logp.numpy()) # numpy to turn off gradient tracking
            loss = tf.add_n([loss_env * tf.reduce_sum(tf.boolean_mask(logp, mask)) for mask, loss_env in results])
            grads = tape.gradient(loss, model.trainable_weights)
            grads = [tf.clip_by_value(grad, -1., 1.) for grad in grads]
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
