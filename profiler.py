import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

cache = {}

def prof_comp(list, index):
    import tensorflow as tf


def prof_link(a, b, size):
    pass

# whether this script should be run in an independent process is not decided
# independent: they are natually decoupled, the result can be for to several runs
# not independent: they share the same tf session which is nice

# format: most parameters (type (comp/link) and input sizes) are in the URL
# the body only contains graphdef (with a special name profilee), which include op and device (and don't forget attr)
# just ignore variable-related ops? (Assign*, *Apply*, Variable*, https://github.com/tensorflow/tensorflow/blob/r1.14/tensorflow/python/framework/graph_util_impl.py#L35)

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        pass

try:
    # parse commandline to get device information
    HTTPServer(('0.0.0.0', 3907), Handler).serve_forever()
except KeyboardInterrupt:
    print("bye~")

# TODO: concurrent profiling if the devices do not overlap? I don't know if tf sessions are thread-safe
