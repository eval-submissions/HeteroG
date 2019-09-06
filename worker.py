import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

def serve_tf(list, index):
    import tensorflow as tf

    tf.distribute.Server(tf.train.ClusterSpec({
        "tge": list
    }), job_name='tge', task_index=index).join()

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
            [index, *list] = args
            index = int(index)
            list = [x.replace('%3A', ':') for x in list]
            serve_tf(list, index)
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
