import tensorflow as tf
import urllib
import time
import os
import json
import numpy as np
def build_model_fn_optimizer():
    """Simple model_fn with optimizer."""
    # TODO(anjalisridhar): Move this inside the model_fn once OptimizerV2 is
    # done?
    optimizer = tf.train.GradientDescentOptimizer(0.2)

    def model_fn(features, labels, mode):  # pylint: disable=unused-argument
        """model_fn which uses a single unit Dense layer."""
        # You can also use the Flatten layer if you want to test a model without any
        # weights.
        num_classes = 1000
        print(features.shape)
        print(labels.shape)
        print("aaaaa")
        from tensorflow.contrib.slim.nets import vgg

        output, _ = vgg.vgg_19(features, 1000)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {"logits": output}
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        loss=tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=output)
        loss = tf.reduce_sum(loss)

        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=loss)

        assert mode == tf.estimator.ModeKeys.TRAIN

        global_step = tf.train.get_or_create_global_step()
        train_op = optimizer.minimize(loss=loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    return model_fn



tf.logging.set_verbosity(tf.logging.INFO)
def setup_workers(workers, protocol="grpc"):
    param = '/'.join(server.replace(':', '%3A') for server in workers)
    for task_id, server in enumerate(workers):
        if task_id == 0: continue
        url = "http://{}:3905/{}/restart/{}/{}/{}".format(server.split(':')[0], int(time.time()) + 10, protocol, task_id, param)
        assert urllib.request.urlopen(url).read() == b'ok'
    time.sleep(1)

config_dict =dict()
if os.path.exists("config.txt"):
    with open("config.txt", "r") as f:
        config_dict = json.load(f)

devices=[
    "/job:worker/replica:0/task:0/device:GPU:0",
    "/job:worker/replica:0/task:0/device:GPU:1",
    "/job:worker/replica:0/task:1/device:GPU:0",
    "/job:worker/replica:0/task:1/device:GPU:1",
    "/job:worker/replica:0/task:2/device:GPU:0",
    "/job:worker/replica:0/task:2/device:GPU:1"

]

#workers = ["10.28.1.26:3901", "10.28.1.25:3901","10.28.1.24:3901","10.28.1.17:3901","10.28.1.16:3901"]
workers = ["10.28.1.26:3901","10.28.1.17:3901","10.28.1.16:3901"]
os.environ["TF_CONFIG"] = '{ "cluster": { "worker": ["10.28.1.26:3901","10.28.1.17:3901","10.28.1.16:3901"]  }, "task": {"type": "worker", "index": 0} }'
setup_workers(workers, "grpc")
# devices=["/job:chief/task:0"]
communication = "nccl"
num_gpus_per_worker=6

cross_op = tf.contrib.distribute.MultiWorkerAllReduce(devices, num_gpus_per_worker=num_gpus_per_worker,
                                                      all_reduce_spec=(communication, 1, -1))
# cross_op = tf.contrib.distribute.AllReduceCrossTowerOps("hierarchical_copy")
# cross_op = tf.contrib.distribute.AllReduceCrossDeviceOps("hierarchical_copy")
distribution=tf.contrib.distribute.MirroredStrategy(num_gpus_per_worker=num_gpus_per_worker,cross_device_ops=cross_op) if num_gpus_per_worker>1 else None
#distribution = tf.contrib.distribute.MirroredStrategy(num_gpus_per_worker=FLAGS.num_clones,
#                                                      cross_tower_ops=cross_op)
# distribution = tf.contrib.distribute.CollectiveAllReduceStrategy(
# num_gpus_per_worker=FLAGS.num_clones)
# distribution=None
# distribution=tf.contrib.distribute.MirroredStrategy(num_gpus_per_worker=FLAGS.num_clones) if FLAGS.num_clones>1 else None
#distribution = tf.distribute.experimental.MultiWorkerMirroredStrategy(
#    tf.distribute.experimental.CollectiveCommunication.NCCL)
config = tf.estimator.RunConfig(
    train_distribute=distribution,
    model_dir="estimator/",
    save_summary_steps=50,
    save_checkpoints_steps=1000,
    log_step_count_steps=1,
    protocol=None)

def train_input_fn():
    train_examples = np.random.sample((100,244,244,3))
    train_labels = np.random.sample((100,1000))
    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    train_dataset = train_dataset.repeat().batch(48)
    return train_dataset

def eval_input_fn():
    train_examples = np.random.sample((100,244,244,3))
    train_labels = np.random.sample((100,1000))
    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    train_dataset = train_dataset.repeat().batch(48)
    return train_dataset

log_hook = tf.train.LoggingTensorHook({"global_step": "global_step"}, every_n_iter=1)
estimator = tf.estimator.Estimator(
    model_fn=build_model_fn_optimizer(), model_dir="estimator/", config=config)
train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=100000)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


