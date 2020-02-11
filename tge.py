import re
import ctypes

PROFILER_T = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32)

libtge = ctypes.cdll.LoadLibrary("./libtge.so")

libtge.tge.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint32]
libtge.tge.restype = ctypes.c_void_p

libtge.topology.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_char_p, ctypes.c_uint32]
libtge.topology.restype = ctypes.c_void_p

libtge.set_option.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
libtge.set_option.restype = ctypes.c_void_p

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

libtge.evaluate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
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
    def __init__(self, graph_def, device_list, sinks=["GradientDescent"]):
        self.sinks = sinks
        self.graph_def = graph_def
        self.devices = device_list
        self.options = {}
        self.flag = 0x03

    def get_result(self):
        return self.graph_def

    @chain
    def compile(self):
        graph_raw = self.graph_def.SerializeToString()
        devices_raw = ' '.join(self.devices).encode('ascii')
        sinks_raw = ' '.join(self.sinks).encode('ascii')
        tge = libtge.tge(self.strategy, self.get_topology(), graph_raw, len(graph_raw), devices_raw, len(devices_raw), sinks_raw, len(sinks_raw))
        for name, value in self.options.items():
            name_raw = str(name).encode('ascii')
            value_raw = str(value).encode('ascii')
            libtge.set_option(tge, name_raw, len(name_raw), value_raw, len(value_raw))
        size = libtge.compile(tge, self.flag)
        buf = ctypes.create_string_buffer(size)
        libtge.read_and_destroy(tge, buf)
        self.graph_def.Clear() # I'm not sure if this line is needed
        self.graph_def.ParseFromString(buf.raw)

    def evaluate(self, profile_dict, trace_path=""):
        graph_raw = self.graph_def.SerializeToString()
        devices_raw = ' '.join(self.devices).encode('ascii')
        profile_raw = ''
        for (name, nreplica), times in profile_dict.items():
            profile_raw += ' '.join([name, str(nreplica), *map(str, times)]) + '\n'
        profile_raw = profile_raw.encode('ascii')
        sinks_raw = ' '.join(self.sinks).encode('ascii')
        tge = libtge.tge(self.strategy, self.get_topology(), graph_raw, len(graph_raw), devices_raw, len(devices_raw), sinks_raw, len(sinks_raw))
        for name, value in self.options.items():
            name_raw = str(name).encode('ascii')
            value_raw = str(value).encode('ascii')
            libtge.set_option(tge, name_raw, len(name_raw), value_raw, len(value_raw))
        size = libtge.compile(tge, self.flag | 0x08)
        trace_path = trace_path.encode('ascii')
        memory = (ctypes.c_uint64 * len(self.devices))(*[0 for x in self.devices])
        result = libtge.evaluate(tge, profile_raw, len(profile_raw), trace_path, len(trace_path), memory)
        buf = ctypes.create_string_buffer(size)
        libtge.read_and_destroy(tge, buf)
        self.graph_def.Clear()
        self.graph_def.ParseFromString(buf.raw)
        return result, list(memory)

    @chain
    def destructify_names(self):
        self.flag = self.flag | 0x04

    @chain
    def set_topology(self, links, paths):
        '''
        links: an array contains the bandwidth of each link. The unit is bytes/time where time is the same unit of profiling
        paths: an array where the i*n+j element is an array of link indexes that in the path of i->j.
        '''
        links_raw = ' '.join(map(str, links)).encode('ascii')
        paths_raw = '\n'.join((' '.join(map(str, path)) for path in paths))
        paths_raw = (paths_raw + '\n').encode('ascii')
        self.topology = libtge.topology(links_raw, len(links_raw), paths_raw, len(paths_raw))

    @chain
    def set_bandwidth(self, intra, inter):
        'conveient method for setting a topology that devices on the same task are independently connected, while devices on different tasks shares a unique link'
        task_map = { device: re.findall(r"task:(\d+)/", device) for device in self.devices }
        links, paths = [inter], [] # the 0-th link is the shared inter link, others are intra links
        for i in range(len(self.devices)):
            for j in range(len(self.devices)):
                if i == j: # the same node, no link needed
                    paths.append([])
                elif task_map[self.devices[i]] == task_map[self.devices[j]]: # intra link
                    paths.append([len(links)])
                    links.append(intra)
                else: # inter link
                    paths.append([0])
        self.set_topology(links, paths)

    @chain
    def replace_placeholder(self, batchsize):
        self.options["replace_placeholder"] = batchsize

    def get_topology(self):
        'default topology use a single shared 100k bytes per micro second bandwidth'
        if hasattr(self, 'topology'):
            return self.topology

        links_raw = '1000000'.encode('ascii')
        paths_raw = '\n'.join('' if i == j else '0' for i in range(len(self.devices)) for j in range(len(self.devices)))
        paths_raw = (paths_raw + '\n').encode('ascii') # since the last element is '', where will be one line missing
        return libtge.topology(links_raw, len(links_raw), paths_raw, len(paths_raw))

    @chain
    def verbose(self):
        self.options["log_forms"] = True
        self.options["log_groups"] = True

    @chain
    def use_nccl(self):
        self.options["use_nccl"] = True

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
    def custom(self, decisions): # each value in decision is an array, where the first element is 0 or 1 indicating PS or all-reduce, followed by the devices
        decision_raw = ''
        for name, decision in decisions.items():
            decision_raw += name + ' ' + str(decision[0])
            for i, j in enumerate(decision[1:]):
                while j > 0:
                    decision_raw += ' ' + str(i)
                    j -= 1
            decision_raw += '\n'
        # print(decision_raw)
        self.strategy = libtge.custom(decision_raw.encode('ascii'), len(decision_raw.encode('ascii')))

    @chain
    def heft(self, profiler):
        self._profiler = PROFILER_T(profiler) # hold the reference to prevent it from being recycled
        self.strategy = libtge.heft(self._profiler)

    @chain
    def not_at_all(self):
        self.strategy = libtge.not_at_all()
