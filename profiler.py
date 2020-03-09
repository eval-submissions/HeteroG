import numpy as np
import tensorflow as tf

class Profiler:
    def __init__(self, graph_def, target=None, sinks=["GradientDescent"]):
        self.graph_def = graph_def
        self.names = { node.name for node in graph_def.node }
        self.sinks = sinks
        self.target = target
        self.profiled = set()
        self.cache = {} # TODO: persistence? LRU?

    def _profile(self, device, run_meta):
        if run_meta is None:
            tf.reset_default_graph()
            tf.import_graph_def(self.graph_def)
            graph = tf.get_default_graph()
            for op in graph.get_operations():
                op._set_device(device)
            init = graph.get_operation_by_name("import/init")

            sess = tf.Session(self.target)#, config=tf.ConfigProto(allow_soft_placement=False))
            sess.run(init)

            placeholders = (node.outputs[0] for node in graph.get_operations() if node.node_def.op == 'Placeholder')
            input_dict = { p: np.random.rand(*p.shape.as_list()) for p in placeholders }

            run_meta = tf.compat.v1.RunMetadata()
            run_opt = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)#, output_partition_graphs=True)
            opt = [graph.get_operation_by_name('import/' + x) for x in self.sinks]
            sess.run(opt, feed_dict=input_dict)
            sess.run(opt, options=run_opt, run_metadata=run_meta, feed_dict=input_dict)

        result = {}
        for dev in run_meta.step_stats.dev_stats:
            if 'Kernel' not in dev.device and 'stream' not in dev.device: # TODO: if no GPU data for this op, use the CPU data
                continue
            for node in dev.node_stats:
                name = node.node_name.split(':')[0]
                if name[:7] == 'import/':
                    name = name[7:]
                if name not in result:
                    result[name] = [float('inf'), 0]
                result[name][0] = min(result[name][0], node.all_start_micros)
                result[name][1] = max(result[name][1], node.all_start_micros + node.all_end_rel_micros)

        for name, [start, end] in result.items():
            self.cache[(name, device)] = end - start

        self.profiled.add(device)

    def profile(self, node_name, device, run_meta=None):
        if device not in self.profiled:
            self._profile(device, run_meta)
        return self.cache.get((node_name, device), 0)
