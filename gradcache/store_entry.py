import numpy as np

from .parameter_wrapper import parameter_wrapper, sift_parameters

from .node import Node


class memodict(collections.OrderedDict):
    """Keep a size limited cache of function results
    Also optionally tracks time and memory usage for the function calls
    """

    def __init__(
        self,
        f,
        maxsize=1,
        enabled=True,
        sample_time=True,
        sample_mem=True,
        track_time=False,
        track_mem=False,
    ):
        collections.OrderedDict.__init__(self)
        self.f = f
        self.maxsize = maxsize
        self.enabled = True

        self.accesses = 0
        self.accesses_weighted = 0

        self.sample_time = sample_time
        self.track_time = track_time
        self.time_samples = []

        self.sample_mem = sample_mem
        self.track_mem = track_mem
        self.mem_samples = []

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def clear(self):
        super().clear()
        self.accesses = 0
        self.accesses_weighted = 0

    def set_size(self, size):
        self.maxsize = size
        while len(self) >= max(self.maxsize, 1):
            self.popitem(last=False)

    def get_state(self):
        return (
            self.accesses,
            self.accesses_weighted,
            np.sum(self.time_samples) / len(self.time_samples),
            np.sum(self.mem_samples) / len(self.mem_samples),
        )

    def set_function(self, f):
        self.f = f
        while len(self) >= max(self.maxsize, 1):
            self.popitem(last=False)
        self.time_samples = []
        self.mem_samples = []

    def add_time(self, t):
        self.time_samples.append(t)

    def add_mem(self, m):
        self.mem_samples.append(m)

    def __getitem__(self, key, extra=None):
        self.accesses += 1
        self.accesses_weighted += 1.0 / max(self.maxsize, 1)
        if super().__contains__(key):
            return super().__getitem__(key)
        while len(self) >= max(self.maxsize, 1):
            self.popitem(last=False)
        mem_sample = (not len(self.mem_samples) and self.sample_mem) or self.track_mem
        time_sample = (
            not len(self.time_samples) and self.sample_time
        ) or self.track_time
        default = (not mem_sample) and (not time_sample)
        if default:
            ret = self.f(key, extra)
        elif time_sample and mem_sample:
            import os
            import psutil

            process = psutil.Process(os.getpid())
            mem0 = process.memory_info().rss
            tic = time.perf_counter()
            ret = self.f(key, extra)
            toc = time.perf_counter()
            mem1 = process.memory_info().rss
            self.add_time(toc - tic)
            self.add_mem(mem1 - mem0)
        elif time_sample:
            tic = time.perf_counter()
            ret = self.f(key, extra)
            toc = time.perf_counter()
            self.add_time(toc - tic)
        elif mem_sample:
            import os
            import psutil

            process = psutil.Process(os.getpid())
            mem0 = process.memory_info().rss
            ret = self.f(key, extra)
            process = psutil.Process(os.getpid())
            mem1 = process.memory_info().rss
            self.add_mem(mem1 - mem0)

        if self.enabled and len(self) < self.maxsize:
            super().__setitem__(key, ret)
        return ret

    def __call__(self, key, extra):
        return self.__getitem__(key, extra)


class entry_context:
    """The context of a function within the larger computation/dependency graph"""

    def __init__(
        self, name, the_store, dependents=None, physical_props=None, props=None
    ):

        self.name = name
        self.the_store = the_store
        self.init_deps = False
        self.init_physical_deps = False
        self.init_implicit_deps = False

        if dependents is None:
            dependents = []
        else:
            self.add_dependencies(dependents)

        self.physical_props = []
        self.props = []
        self.physical_props_indices = []
        self.props_indices = []
        if props is not None:
            if physical_props is not None:
                self.add_physical_dependencies(props, physical_props)
            else:
                raise ValueError("Need both props and physical_props or neither")
        else:
            if physical_props is not None:
                raise ValueError("Need both props and physical_props or neither")

        self.implicit_physical_props = []

    def add_dependencies(self, dependents):
        if self.init_deps:
            raise RuntimeError("Dependencies already initialized!")
        if dependents is None:
            dependents = []
        self.dependents = dependents
        self.init_deps = True

    def add_physical_dependencies(self, props, physical_props):
        if self.init_physical_deps:
            raise RuntimeError("Physical dependencies already initialized!")
        if not self.init_deps:
            raise RuntimeError(
                "Dependencies must be initialized before initializing physical dependencies!"
            )
        if self.dependents is not None:
            these_deps = set(self.dependents)
        else:
            these_deps = set()
        these_physical_props = these_deps - props
        these_props = these_deps - these_physical_props

        for i, name in enumerate(self.dependents):
            if name in these_physical_props:
                self.physical_props.append(name)
                self.physical_props_indices.append(i)
            elif name in these_props:
                self.props.append(name)
                self.props_indices.append(i)
            else:
                raise ValueError("Not in props or physical_props:", name)

    def add_implicit_dependencies(self, prop_map):
        if (not self.init_deps) or (not self.init_physical_deps):
            raise RuntimeError(
                "Dependencies and physical dependencies must be initialized before initializing implicit dependencies!"
            )

        if len(self.implicit_physical_props) == 0:
            prop_deps = set()
            for dprop in initialized_cache_entry.props:
                dprop_ = prop_map[dprop]
                dprop_.add_implicit_dependencies(prop_map)
                prop_deps |= set(dprop_.physical_props)
                prop_deps |= set(dprop_.implicit_physical_props)
            prop_deps -= set(self.physical_props)
            self.implicit_physical_props = list(prop_deps)

    def extract_params(self, physical_parameters):
        these_params = tuple([physical_parameters[k] for k in self.physical_props])
        these_params += tuple(
            [physical_parameters[k] for k in self.implicit_physical_props]
        )
        return these_params

    def extract_values(self, parameters, physical_parameters):
        values = [None for i in range(len(self.dependents))]
        for v, i in zip(parameters, self.physical_props_indices):
            values[i] = v
        for prop, i in zip(self.props, self.props_indices):
            values[i] = self.the_store.get_prop(prop, physical_parameters)
        return values

    def compute(self, parameters, physical_parameters, *args, **kwargs):
        values = self.extract_values(parameters, physical_parameters)
        return self.atomic_operation(*values)

    def __call__(self, physical_parameters, *args, **kwargs):
        these_params = self.extract_params(physical_parameters)
        return self.compute(these_params, physical_parameters, *args, **kwargs)


class function_wrapper:
    def __init__(self, function, name, arg_names):
        self.function = function
        self.arg_names = arg_names
        self.context = None
        self.cache = memodict(f, 1)

    def get_cache_state(self):
        return self.cache.get_state()

    def get_cache_item(self, key):
        pass

    def compute_item(self, key):
        pass

    def enable_cache(self):
        self.cache.enable()

    def disable_cache(self):
        self.cache.disable()

    def clear_cache(self):
        self.cache.clear()

    def set_cache_size(self, size):
        self.cache.set_size(size)

    def set_context(self, context):
        self.context = context

    # basic numerical evaluation
    def eval(self, parameters):
        args = [parameters[k] for k in self.arg_names]
        return self.function(*args)

    # Evaluation with gradient tracking
    # Accepts only parameter_wrappers
    def eval_grad(self, parameter_wrappers):
        nodes = [
            Node(name, [], value=pwrap)
            for name, pwrap in zip(self.arg_names, parameter_wrappers)
        ]
        res_node = self.function(*nodes)
        return res_node.value

    def __call__(self, *args):
        new_args = []
        have_grad = False
        for name, arg in zip(self.arg_names, args):
            if isinstance(arg, parameter_wrapper):
                have_grad = True
            else:
                arg = parameter_wrapper(name, arg)
            new_args.append(arg)
        if have_grad:
            return self.eval_grad(new_args)
        else:
            return self.eval(args)
