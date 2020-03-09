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

libtge.create_target.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32] * 4
libtge.create_target.restype = ctypes.c_void_p

libtge.destroy_target.argtypes = [ctypes.c_void_p]
libtge.destroy_target.restype = None

libtge.compute_size.argtypes = [ctypes.c_void_p]
libtge.compute_size.restype = ctypes.c_uint32

libtge.read_protobuf.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
libtge.read_protobuf.restype = None

libtge.create_editor.argtypes = [ctypes.POINTER(ctypes.c_char), ctypes.c_uint32]
libtge.create_editor.restype = ctypes.c_void_p

libtge.destroy_editor.argtypes = [ctypes.c_void_p]
libtge.destroy_editor.restype = None

libtge.compile.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
libtge.compile.restype = None

libtge.evaluate.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(ctypes.c_char), ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint64)]
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
        self.graph_proto_type = type(graph_def)
        self.options = {}

        graph_raw = graph_def.SerializeToString()
        self.graph = libtge.create_graph(graph_raw, len(graph_raw))

        # default topology
        self.links = [1000000]
        self.paths = [[] if i == j else [0] for i in range(len(device_list)) for j in range(len(device_list))]

        self.editor = None
        self.target = None
        self.compiled = False # if the target is compiled. Being True also implies that self.target is not None.

    def __del__(self):
        libtge.destroy_graph(self.graph)

        if self.editor is not None:
            libtge.destroy_editor(self.editor)

        if self.target is not None:
            libtge.destroy_target(self.target)

    def get_result(self):
        assert self.target is not None
        size = libtge.compute_size(self.target)
        buf = ctypes.create_string_buffer(size)
        libtge.read_protobuf(self.target, buf)
        result = self.graph_proto_type()
        result.ParseFromString(buf.raw)
        return result

    def get_groups(self):
        names_raw = ' '.join((node.name for node in self.graph_def.node)).encode('ascii')
        result = (ctypes.c_uint32 * len(self.graph_def.node))(*(0 for x in self.graph_def.node))
        libtge.get_groups(self.graph, names_raw, len(names_raw), result)
        return list(result)

    @chain
    def compile(self):
        assert self.editor is not None
        self._set_options()
        self._create_target()
        libtge.compile(self.graph, self.editor, self.target)
        self.compiled = True

        # for backward compatibility
        self.remove_collocation_hint()
        self.remove_shape_hint()

    def evaluate(self, profile_dict, trace_path=""):
        if not self.compiled: # for backward compatibility
            self.compile()
        self.remove_dangling_nodes()

        profile_raw = ''
        for (name, nreplica), times in profile_dict.items():
            profile_raw += ' '.join([name, str(nreplica), *map(str, times)]) + '\n'
        profile_raw = profile_raw.encode('ascii')
        trace_path = trace_path.encode('ascii')
        memory = (ctypes.c_uint64 * len(self.devices))(*(0 for x in self.devices))
        result = libtge.evaluate(self.target, profile_raw, len(profile_raw), trace_path, len(trace_path), memory)
        return result, list(memory)

    def _set_options(self):
        for name, value in self.options.items():
            name_raw = str(name).encode('ascii')
            value_raw = str(value).encode('ascii')
            libtge.set_option(self.graph, name_raw, len(name_raw), value_raw, len(value_raw))

    def _create_target(self):
        devices_raw = ' '.join(self.devices).encode('ascii')
        sinks_raw = ' '.join(self.sinks).encode('ascii')
        links_raw = ' '.join(map(str, self.links)).encode('ascii')
        paths_raw = '\n'.join((' '.join(map(str, path)) for path in self.paths))
        paths_raw = (paths_raw + '\n').encode('ascii')
        if self.target is not None:
            libtge.destroy_target(self.target)
        self.target = libtge.create_target(
            devices_raw, len(devices_raw),
            links_raw, len(links_raw),
            paths_raw, len(paths_raw),
            sinks_raw, len(sinks_raw)
        )
        self.compiled = False

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

    @chain
    def verbose(self):
        self.options["log_forms"] = True
        self.options["log_groups"] = True

    @chain
    def use_nccl(self):
        self.options["allreduce_implementation"] = 'nccl'

    @chain
    def use_collective(self):
        self.options["allreduce_implementation"] = 'collective'

    @chain
    def custom(self, decisions): # for backward compatibility
        self.set_strategy(decisions)

    @chain
    def set_strategy(self, strategy): # each value in decision is an array, where the first element is 0 or 1 indicating PS or all-reduce, followed by the devices
        decision_raw = ''
        for name, decision in strategy.items():
            decision_raw += name + ' ' + str(decision[0])
            for i, j in enumerate(decision[1:]):
                while j > 0:
                    decision_raw += ' ' + str(i)
                    j -= 1
            decision_raw += '\n'
        if self.editor is not None:
            libtge.destroy_editor(self.editor)
        self.editor = libtge.create_editor(decision_raw.encode('ascii'), len(decision_raw.encode('ascii')))
