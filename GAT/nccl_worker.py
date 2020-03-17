import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from tensorflow.distribute.cluster_resolver import TFConfigClusterResolver
import json
def serve_tf(list, index, protocol):
    import tensorflow as tf
    clus = dict()
    clus["cluster"] = {"worker":list}
    clus["task"] = {"type":"worker","index":index}
    os.environ["TF_CONFIG"] = json.dumps(clus)
    resolver = TFConfigClusterResolver()
    cluster = resolver.cluster_spec()
    dist = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        tf.distribute.experimental.CollectiveCommunication.NCCL)
    config = dist.update_config_proto(tf.ConfigProto())
    config.ClearField("device_filters")
    config.allow_soft_placement = True  # log_device_placement=True)
    config.gpu_options.allow_growth = True
    tf.distribute.Server(cluster, job_name='worker', task_index=index, protocol=protocol,config=config).join()

pid = 0

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        global pid
        print(self.path)

        try:
            [_, timestamp, cmd, *args] = self.path.split('/')
        except:
            return print("wrong path")

        if int(timestamp) < int(time.time()):
            return print("expired request")

        if cmd != 'restart':
            return print("unknown command")

        if pid != 0:
            os.kill(pid, 9)

        pid = os.fork()
        if pid == 0:
            [protocol, index, *list] = args
            index = int(index)
            list = [x.replace('%3A', ':') for x in list]
            serve_tf(list, index, protocol)
        else:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'ok')

try:
    HTTPServer(('0.0.0.0', 3905), Handler).serve_forever()
except KeyboardInterrupt:
    if pid != 0:
        os.kill(pid, 9)
    print("bye~")
