import tensorflow as tf
import numpy as np
from environment import sample, evaluate, sample_and_evaluate
from utils import info

def search(record, c_embeddings, t_embeddings):
    c_var = tf.Variable(initial_value=c_embeddings, trainable=True)
    t_var = tf.Variable(initial_value=t_embeddings, trainable=True)
    variables = [c_var, t_var]

    optimizer = tf.keras.optimizers.Adam(learning_rate=.01, clipnorm=1)
    L2_regularization_factor = .00001

    hist = []
    for epoch in range(2000):
        with tf.GradientTape() as tape:
            tape.watch(variables)
            logit = tf.matmul(c_var, t_var, transpose_b=True)

            # info(logit)

            strategy = sample(logit)
            sqrt_time, oom, leftout = evaluate(record, [1] * strategy.shape[0], strategy)

            if 'best' not in record or record['best'] > sqrt_time:
                record['best'] = sqrt_time

            if 'dp' not in record:
                record['dp'] = evaluate(record, [1] * strategy.shape[0], np.ones(strategy.shape, int))[0]

            hist.append(sqrt_time)
            hist = hist[-100:]
            baseline = np.mean(hist)
            advantage = -(sqrt_time - baseline) / baseline

            loss = advantage * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(strategy.astype(np.float32), logit))

            info(loss.numpy(), record['best'], record['dp'])

            if L2_regularization_factor > 0:
                for v in variables:
                    loss += L2_regularization_factor * tf.nn.l2_loss(v)

            grads = tape.gradient(loss, variables)
            # info([tf.reduce_mean(tf.abs(grad)).numpy() for grad in grads])
            optimizer.apply_gradients(zip(grads, variables))

