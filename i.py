import sys
import tensorflow as tf
from tensorflow.distribute.cluster_resolver import TFConfigClusterResolver
from tensorflow.python.ops import collective_ops
tf.logging.set_verbosity('DEBUG')

def test_dist():
    resolver = TFConfigClusterResolver()
    cluster = resolver.cluster_spec()

    dist = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.NCCL)

    sess_config = dist.update_config_proto(tf.ConfigProto())
    sess_config.ClearField("device_filters")

    print(sess_config)

    server = tf.distribute.Server(
        cluster, job_name="worker", task_index=0, config=sess_config)

    print('num replicas', dist.num_replicas_in_sync)

    ts = []
    for task_id in (0, 1):
        with tf.device('/job:worker/task:{0}/device:GPU:0'.format(task_id)):
            t = tf.Variable([1.0,3.0*task_id], dtype=tf.float32, name='myvar')
            ts.append(t)

    with dist.scope():
        with tf.device('/job:worker/task:0/device:GPU:0'):
            sum0 = collective_ops.all_reduce(t[0], 2, 0, 1, 'Add', 'Id')
        with tf.device('/job:worker/task:1/device:GPU:0'):
            sum1 = collective_ops.all_reduce(t[1], 2, 0, 1, 'Add', 'Id')

        with tf.device('/job:worker/task:0/device:GPU:0'):
            sumb0 = collective_ops.all_reduce(sum1, 2, 0, 2, 'Add', 'Id')
        with tf.device('/job:worker/task:1/device:GPU:0'):
            sumb1 = collective_ops.all_reduce(sum0, 2, 0, 2, 'Add', 'Id')

        sess = tf.compat.v1.Session(server.target, config=sess_config)
        sess.run(tf.compat.v1.global_variables_initializer())

        print('tensor value', sess.run([sumb0, sumb1]))

    with open("graph_def", "w") as f:
        f.write(str(tf.get_default_graph().as_graph_def()))

if int(sys.argv[1]) == 0:
    test_dist()
else:
    resolver = TFConfigClusterResolver()
    cluster = resolver.cluster_spec()
    dist = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.NCCL)
    sess_config = dist.update_config_proto(tf.ConfigProto())
    sess_config.ClearField("device_filters")
    server = tf.distribute.Server(
        cluster, job_name="worker", task_index=1, config=sess_config)
    server.join()
