import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

import os, json

BUFFER_SIZE = 10000
BATCH_SIZE = 64

def input_fn(mode, input_context=None):
  datasets, info = tfds.load(name='mnist',
                                with_info=True,
                                as_supervised=True)
  mnist_dataset = (datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else
                   datasets['test'])

  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  if input_context:
    mnist_dataset = mnist_dataset.shard(input_context.num_input_pipelines,
                                        input_context.input_pipeline_id)
  return mnist_dataset.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

import sys

os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["127.0.0.1:3901", "127.0.0.1:3902"]
    },
    'task': {'type': 'worker', 'index': int(sys.argv[1])}
})

LEARNING_RATE = 1e-4
def model_fn(features, labels, mode):
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  logits = model(features, training=False)
  labels = tf.one_hot(labels, 10)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {'logits': logits}
    return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

  optimizer = tf.compat.v1.train.GradientDescentOptimizer(
      learning_rate=LEARNING_RATE)
  loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
  loss = tf.reduce_sum(loss) * (1. / BATCH_SIZE)
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode, loss=loss)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=optimizer.minimize(
          loss, tf.compat.v1.train.get_or_create_global_step()))

# strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(tf.distribute.experimental.CollectiveCommunication.NCCL)

communication="nccl"
cross_op = tf.contrib.distribute.MultiWorkerAllReduce(["/job:worker/task:0","/job:worker/task:1"],num_gpus_per_worker=1,all_reduce_spec=(communication,1,-1))
strategy = tf.contrib.distribute.MirroredStrategy(num_gpus_per_worker=1,cross_tower_ops=cross_op)

config = tf.estimator.RunConfig(train_distribute=strategy)

classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir='/tmp/multiworker', config=config)
tf.estimator.train_and_evaluate(
    classifier,
    train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
    eval_spec=tf.estimator.EvalSpec(input_fn=input_fn)
)
