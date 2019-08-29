def model_fn():
    x = tf.placeholder(tf.float32, shape=(None, 1024))
    y = tf.placeholder(tf.float32, shape=(None, 10,))
    hidden = tf.contrib.slim.fully_connected(x, 256, activation_fn=tf.nn.softmax)
    output = tf.contrib.slim.fully_connected(hidden, 10, activation_fn=tf.nn.softmax)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output)
    optimizer = tf.train.GradientDescentOptimizer(0.2).minimize(tf.reduce_sum(loss))

    return optimizer
