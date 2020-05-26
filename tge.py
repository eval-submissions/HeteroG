import re
import ctypes

PROFILER_T = ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32)

libtge = ctypes.cdll.LoadLibrary("./libtge.so")

libtge.create_graph.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
libtge.create_graph.restype = ctypes.c_void_p

libtge.destroy_graph.argtypes = [ctypes.c_void_p]
libtge.destroy_graph.restype = None

libtge.set_option.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
libtge.set_option.restype = None

libtge.get_groups.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32)]
libtge.get_groups.restype = None

libtge.edit_graph.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
libtge.edit_graph.restype = None

libtge.reset_graph.argtypes = [ctypes.c_void_p]
libtge.reset_graph.restype = None

libtge.create_target.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32] * 5
libtge.create_target.restype = ctypes.c_void_p

libtge.destroy_target.argtypes = [ctypes.c_void_p]
libtge.destroy_target.restype = None

libtge.compute_size.argtypes = [ctypes.c_void_p]
libtge.compute_size.restype = ctypes.c_uint32

libtge.read_protobuf.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
libtge.read_protobuf.restype = None

libtge.compile.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libtge.compile.restype = None

libtge.create_profiler.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
libtge.create_profiler.restype = ctypes.c_void_p

libtge.destroy_profiler.argtypes = [ctypes.c_void_p]
libtge.destroy_profiler.restype = None

libtge.heft_rank.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libtge.heft_rank.restype = None

libtge.heft_control.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
libtge.heft_control.restype = None

libtge.evaluate.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
libtge.evaluate.restype = ctypes.c_uint64

libtge.remove_collocation_hint.argtypes = [ctypes.c_void_p]
libtge.remove_collocation_hint.restype = None

libtge.remove_shape_hint.argtypes = [ctypes.c_void_p]
libtge.remove_shape_hint.restype = None

libtge.destruct_names.argtypes = [ctypes.c_void_p]
libtge.destruct_names.restype = None

libtge.remove_dangling_nodes.argtypes = [ctypes.c_void_p]
libtge.remove_dangling_nodes.restype = None

def chain(func):
    def chained(self, *args, **kwargs):
        func(self, *args, **kwargs)
        return self
    return chained

class TGE:
    def __init__(self, graph_def, device_list, sinks=["GradientDescent"]):
        self.sinks = sinks
        self.devices = device_list
        self.graph_def = graph_def

        graph_raw = graph_def.SerializeToString()
        self.graph = libtge.create_graph(graph_raw, len(graph_raw))

        # default topology
        self.links = [1000000]
        self.paths = [[] if i == j else [0] for i in range(len(device_list)) for j in range(len(device_list))]
        self.nccls = {}

        self.strategy = None
        self.target = None
        self.profiler = None
        self.compiled = False # if the target is compiled. Being True also implies that self.target is not None.
        self.edited = False # if the graph is edited. It must be reset before another editing.

    def __del__(self):
        libtge.destroy_graph(self.graph)

        if self.target is not None:
            libtge.destroy_target(self.target)

        if self.profiler is not None:
            libtge.destroy_profiler(self.profiler)

    def get_result(self):
        assert self.target is not None
        size = libtge.compute_size(self.target)
        buf = ctypes.create_string_buffer(size)
        libtge.read_protobuf(self.target, buf)
        result = type(self.graph_def)()
        result.ParseFromString(buf.raw)
        return result

    def get_groups(self):
        names_raw = ' '.join((node.name for node in self.graph_def.node)).encode('ascii')
        result = (ctypes.c_uint32 * len(self.graph_def.node))(*(0 for x in self.graph_def.node))
        libtge.get_groups(self.graph, names_raw, len(names_raw), result)
        return list(result)

    @chain
    def compile(self):
        assert self.strategy is not None
        self._create_target()
        self._edit()
        libtge.compile(self.graph, self.target)
        self.compiled = True

        # for backward compatibility
        self.remove_collocation_hint()
        self.remove_shape_hint()

    @chain
    def heft(self, profile_dict, add_control_dependency=False):
        if not self.compiled:
            self.compile()

        self._create_profiler(profile_dict)
        if add_control_dependency:
            libtge.heft_control(self.target, self.profiler)
        else:
            libtge.heft_rank(self.target, self.profiler)

    def evaluate(self, profile_dict, trace_path=""):
        if not self.compiled: # for backward compatibility
            self.compile()
        self.remove_dangling_nodes()

        trace_path = trace_path.encode('ascii')
        memory = (ctypes.c_uint64 * len(self.devices))(*(0 for x in self.devices))
        self._create_profiler(profile_dict)
        result = libtge.evaluate(self.target, self.profiler, trace_path, len(trace_path), memory)
        self.target = None # evaluator now takes the ownership of target

        return result, list(memory)

    def _create_target(self):
        devices_raw = ' '.join(self.devices).encode('ascii')
        sinks_raw = ' '.join(self.sinks).encode('ascii')
        links_raw = ' '.join(map(str, self.links)).encode('ascii')
        paths_raw = '\n'.join((' '.join(map(str, path)) for path in self.paths))
        paths_raw = (paths_raw + '\n').encode('ascii')
        nccls_raw = '\n'.join((' '.join([k, *map(str, v)]) for k, v in self.nccls.items()))
        nccls_raw = (nccls_raw + '\n').encode('ascii')

        if self.target is not None:
            libtge.destroy_target(self.target)
        self.target = libtge.create_target(
            devices_raw, len(devices_raw),
            links_raw, len(links_raw),
            paths_raw, len(paths_raw),
            sinks_raw, len(sinks_raw),
            nccls_raw, len(nccls_raw)
        )
        self.compiled = False

    def _edit(self):
        strategy_raw = ''
        for name, strategy in self.strategy.items():
            strategy_raw += name + ' ' + str(strategy[0])
            for i, j in enumerate(strategy[1:]):
                while j > 0:
                    strategy_raw += ' ' + str(i)
                    j -= 1
            strategy_raw += '\n'
        strategy_raw = strategy_raw.encode('ascii')
        if self.edited:
            libtge.reset_graph(self.graph)
        libtge.edit_graph(self.graph, self.target, strategy_raw, len(strategy_raw))
        self.edited = True

    def _create_profiler(self, profile_dict):
        profile_raw = ''
        for (name, nreplica), times in profile_dict.items():
            profile_raw += ' '.join([name, str(nreplica), *map(str, times)]) + '\n'
        profile_raw = profile_raw.encode('ascii')
        if self.profiler is not None:
            libtge.destroy_profiler(self.profiler)
        self.profiler = libtge.create_profiler(profile_raw, len(profile_raw))

    @chain
    def remove_collocation_hint(self):
        assert self.compiled
        libtge.remove_collocation_hint(self.target)

    @chain
    def remove_shape_hint(self):
        assert self.compiled
        libtge.remove_shape_hint(self.target)

    @chain
    def destruct_names(self):
        assert self.compiled
        libtge.destruct_names(self.target)

    @chain
    def remove_dangling_nodes(self):
        assert self.compiled
        libtge.remove_dangling_nodes(self.target)

    @chain
    def set_topology(self, links, paths):
        """
        links: an array contains the bandwidth of each link. The unit is bytes/time where time is the same unit of profiling
        paths: an array where the i*n+j element is an array of link indexes that in the path of i->j.
        """
        self.links = links
        self.paths = paths

    @chain
    def set_bandwidth(self, intra, inter):
        """convenient method for setting a topology that devices on the same task are independently connected, while devices on different tasks shares a unique link"""
        task_map = { device: int(re.findall(r"task:(\d+)/", device)[0]) for device in self.devices }
        if type(intra) is not dict: # for backward compatibility
            intra = { task: intra for task in task_map.values() }
        links, paths = [inter], [] # the 0-th link is the shared inter link, others are intra links
        for i in range(len(self.devices)):
            for j in range(len(self.devices)):
                if i == j: # the same node, no link needed
                    paths.append([])
                elif task_map[self.devices[i]] == task_map[self.devices[j]]: # intra link
                    paths.append([len(links)])
                    links.append(intra[task_map[self.devices[i]]])
                else: # inter link
                    paths.append([0])
        self.set_topology(links, paths)

    @chain
    def set_nccl_model(self, model):
        """use profiler.py to make a model"""
        self.nccls = model

    def _set_option(self, name, value):
        name_raw = str(name).encode('ascii')
        value_raw = str(value).encode('ascii')
        libtge.set_option(self.graph, name_raw, len(name_raw), value_raw, len(value_raw))

    @chain
    def replace_placeholder(self, batchsize):
        self._set_option("replace_placeholder", batchsize)

    @chain
    def fill_batchsize(self, batchsize):
        self._set_option("fill_batchsize", batchsize)

    @chain
    def verbose(self):
        self._set_option("log_forms", True)
        self._set_option("log_groups", True)

    @chain
    def use_nccl(self):
        # self._set_option("allreduce_implementation", 'nccl')
        pass

    @chain
    def use_collective(self):
        # self._set_option("allreduce_implementation", 'collective')
        pass

    @chain
    def custom(self, decisions): # for backward compatibility
        self.set_strategy(decisions)

    @chain
    def set_strategy(self, strategy): # each value is an array, where the first element is 0 or 1 indicating PS or all-reduce, followed by the devices
        self.strategy = strategy
