import numpy as np
import tensorflow as tf
import re
import itertools

class NcclProfiler:
    def __init__(self, devices, target, seed=3399):
        self.target = target
        self.seed = seed
        self.devices = {}

        for dev in devices:
            task = re.search("task:(\d+)/", dev)[1]
            if task in self.devices.keys():
                self.devices[task].append(dev)
            else:
                self.devices[task] = [dev]

        for devs in self.devices.values():
            devs.sort()

    def profile(self):
        results = {}

        for task, devs in self.devices.items():
            results[','.join(sorted(devs))] = self._model([x for i in range(5) for x in self._profile(devs)])

        for tasks in (t for i in range(2, len(self.devices)+1) for t in itertools.combinations(self.devices.keys(), i)):
            devs = [self.devices[t][0] for t in tasks] # the first (alphabet order) device is the leader of the task
            results[','.join(sorted(devs))] = self._model([x for i in range(5) for x in self._profile(devs)])

        return results

    def _model(self, data):
        from sklearn.linear_model import HuberRegressor
        model1 = HuberRegressor().fit([[x] for x, y in data if x < 2**8], [y for x, y in data if x < 2**8])
        model2 = HuberRegressor().fit([[x] for x, y in data if x > 2**10], [y for x, y in data if x > 2**10])
        return [model1.coef_[0].item(), model1.intercept_.item(), model2.coef_[0].item(), model2.intercept_.item()]

    def _profile(self, devices):
        from tensorflow.python.ops import collective_ops

        id = self.seed
        self.seed += 1

        result = []
        for size in (2**i for i in range(21)): # 1 KB to 1GB
            handles = []
            tf.reset_default_graph()
            for dev in devices:
                with tf.device(dev):
                    x = tf.random.uniform((size, 128), dtype=tf.dtypes.float64)
                    nccl = collective_ops.all_reduce(x, len(devices), id, id, 'Add', 'Id')
                    handles.append(tf.identity(nccl))
            run_meta = tf.compat.v1.RunMetadata()
            run_opt = tf.compat.v1.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            sess = tf.Session(self.target)
            sess.run(handles)
            sess.run(handles, options=run_opt, run_metadata=run_meta)

            time = min(node.all_end_rel_micros for d in run_meta.step_stats.dev_stats for node in d.node_stats if 'CollectiveReduce' in node.node_name)
            result.append((size, time))

        return result

class Profiler:
    def __init__(self, graph_def, batchsize, target=None, sinks=["GradientDescent"]):
        self.graph_def = graph_def
        self.batchsize = batchsize
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
            input_dict = { p: np.random.rand(self.batchsize, *p.shape.as_list()[1:]) for p in placeholders }

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
