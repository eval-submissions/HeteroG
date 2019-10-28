import ctypes

PROFILER_T = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32)

libtge = ctypes.cdll.LoadLibrary("./libtge.so")

libtge.tge.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint32]
libtge.tge.restype = ctypes.c_void_p

libtge.topology.argtypes = [ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint32]
libtge.topology.restype = ctypes.c_void_p

libtge.not_at_all.argtypes = []
libtge.not_at_all.restype = ctypes.c_void_p

# libtge.data_parallel.argtypes = [ctypes.c_byte, ctypes.c_byte]
# libtge.data_parallel.restype = ctypes.c_void_p

# libtge.heft.argtypes = [PROFILER_T]
# libtge.heft.restype = ctypes.c_void_p

libtge.custom.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
libtge.custom.restype = ctypes.c_void_p

libtge.compile.argtypes = [ctypes.c_void_p, ctypes.c_ubyte]
libtge.compile.restype = ctypes.c_uint32

libtge.evaluate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
libtge.evaluate.restype = ctypes.c_uint64

libtge.read_and_destroy.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
libtge.read_and_destroy.restype = None

def chain(func):
    def chained(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self
    return chained

# TODO: ensure the calling order - get result only after compiling which in turn after setting strategy.
# Also, replcaing strategy without compiling causes memory leak on the Rust side.
class TGE:
    def __init__(self, graph_def, device_list):
        self.graph_def = graph_def
        self.devices = device_list
        self.flag = 0x03

    def get_result(self):
        return self.graph_def

    @chain
    def compile(self):
        graph_raw = self.graph_def.SerializeToString()
        device_raw = ' '.join(self.devices).encode('ascii')
        tge = libtge.tge(self.strategy, self.get_topology(), graph_raw, len(graph_raw), device_raw, len(device_raw))
        size = libtge.compile(tge, self.flag)
        buf = ctypes.create_string_buffer(size)
        libtge.read_and_destroy(tge, buf)
        self.graph_def.Clear() # I'm not sure if this line is needed
        self.graph_def.ParseFromString(buf.raw)

    def evaluate(self, profile_dict):
        graph_raw = self.graph_def.SerializeToString()
        device_raw = ' '.join(self.devices).encode('ascii')
        profile_raw = ''
        for name, time in profile_dict.items():
            profile_raw += name + ' ' + str(time) + '\n'
        profile_raw = profile_raw.encode('ascii')
        tge = libtge.tge(self.strategy, self.get_topology(), graph_raw, len(graph_raw), device_raw, len(device_raw))
        size = libtge.compile(tge, self.flag)
        result = libtge.evaluate(tge, profile_raw, len(profile_raw))
        buf = ctypes.create_string_buffer(size)
        libtge.read_and_destroy(tge, buf)
        self.graph_def.Clear()
        self.graph_def.ParseFromString(buf.raw)
        return result

    @chain
    def destructify_names(self):
        self.flag = self.flag | 0x04

    @chain
    def set_topology(self, links, paths):
        '''
        links: an array contains the bandwidth of each link. The unit is bytes/time where time is the same unit of profiling
        paths: an array where the i*n+j element is an array of link indexes that in the path of i->j.
        '''
        links_raw = ' '.join(links).encode('ascii')
        paths_raw = '\n'.join((' '.join(path) for path in paths))
        self.topology = libtge.topology(len(self.devices), links_raw, len(links_raw), paths_raw, len(paths_raw))

    def get_topology(self):
        'default topology use a single shared near-infinity bandwidth if not set'
        if hasattr(self, 'topology'):
            return self.topology

        links_raw = '1000000000'.encode('ascii')
        paths_raw = '0\n' * (len(self.devices) * len(self.devices))
        return libtge.topology(len(self.devices), links_raw, len(links_raw), paths_raw, len(paths_raw))

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
    def custom(self, decisions):
        decision_raw = ''
        for name, decision in decisions.items():
            decision_raw += name + ' ' + str(decision) + '\n'
        self.strategy = libtge.custom(decision_raw.encode('ascii'), len(decision_raw.encode('ascii')))

    @chain
    def heft(self, profiler):
        self._profiler = PROFILER_T(profiler) # hold the reference to prevent it from being recycled
        self.strategy = libtge.heft(self._profiler)

    @chain
    def not_at_all(self):
        self.strategy = libtge.not_at_all()
