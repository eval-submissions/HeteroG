import tensorflow as tf
tf.distribute.Server(tf.train.ClusterSpec({
    "tge": ["net-g10:3901", "net-g11:3901"]
}), job_name='tge', task_index=0).join()
