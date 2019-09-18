import ctypes

libtge = ctypes.cdll.LoadLibrary("./libtge.so")

libtge.tge.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32]
libtge.tge.restype = ctypes.c_void_p

libtge.not_at_all.argtypes = []
libtge.not_at_all.restype = ctypes.c_void_p

libtge.data_parallel.argtypes = [ctypes.c_byte, ctypes.c_byte]
libtge.data_parallel.restype = ctypes.c_void_p

libtge.compile.argtypes = [ctypes.c_void_p]
libtge.compile.restype = ctypes.c_uint32

libtge.read_and_destroy.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
libtge.read_and_destroy.restype = None

def chain(func):
    def chained(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self
    return chained

class TGE:
    def __init__(self):
        self.ctx = None

    @chain
    def set_graph_def(self, graph_def):
        self.graph_def = graph_def

    def get_graph_def(self):
        return self.graph_def

    @chain
    def set_devices(self, devices):
        self.devices = devices

    @chain
    def compile(self):
        graph_raw = self.graph_def.SerializeToString()
        device_raw = ' '.join(self.devices).encode('ascii')
        tge = libtge.tge(self.strategy, graph_raw, len(graph_raw), device_raw, len(device_raw))
        size = libtge.compile(tge)
        print(size)
        buf = ctypes.create_string_buffer(size)
        libtge.read_and_destroy(tge, buf)
        self.graph_def.Clear() # I'm not sure if this line is needed
        self.graph_def.ParseFromString(buf.raw)

    @chain
    def data_parallel(self, method):
        methods_dict = {
            "ps0": 1,
            "ring": 2,
            "nccl": 3
        }

        inner = methods_dict[method.lower()]
        self.strategy = libtge.data_parallel(inner, 0)

    @chain
    def not_at_all(self):
        self.strategy = libtge.not_at_all()
