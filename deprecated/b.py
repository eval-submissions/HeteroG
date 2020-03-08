import sys
import tensorflow as tf
from tensorflow.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.ops import collective_ops
tf.logging.set_verbosity('DEBUG')

def test_dist(task_id):
    resolver = TFConfigClusterResolver()
    cluster = resolver.cluster_spec()

    dist = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.NCCL)

    sess_config = dist.update_config_proto(tf.ConfigProto())
    sess_config.ClearField("device_filters")

    print(sess_config)

    server = tf.distribute.Server(
        cluster, job_name="worker", task_index=task_id, config=sess_config)

    print('num replicas', dist.num_replicas_in_sync)

    with tf.device('/job:worker/task:{0}/device:GPU:0'.format(task_id)):
        t = tf.Variable([1.0,3.0*task_id], dtype=tf.float32, name='myvar')

    with tf.device('/job:worker/task:0/device:GPU:0'):
        t = tf.identity(t)

    with tf.device('/job:worker/task:1/device:GPU:0'):
        t = tf.identity(t)

    with dist.scope():
        # with tf.device('/job:worker/task:{0}/device:GPU:0'.format(task_id)):
        #     all_ts = dist.experimental_run_v2(sum_deltas_fn, args=[t])
        # delta_sums_results = dist.reduce(tf.distribute.ReduceOp.SUM, all_ts)
        delta_sums_results = collective_ops.all_reduce(t, 2, 0, 1, 'Add', 'Id')

        sess = tf.compat.v1.Session(server.target, config=sess_config)
        sess.run(tf.compat.v1.global_variables_initializer())

        print('tensor', delta_sums_results)
        print('tensor value', sess.run(delta_sums_results))

test_dist(int(sys.argv[1]))
