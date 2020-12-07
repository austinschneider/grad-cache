import numpy as np
import collections
import time

try:
    from .parameter_wrapper import parameter_wrapper, sift_parameters
except:
    from parameter_wrapper import parameter_wrapper, sift_parameters

try:
    from .node import Node
except:
    from node import Node


class function_cache(collections.OrderedDict):
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
        self, name, the_store=None, dependents=None, physical_props=None, props=None
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
        self.callback = None

    def set_store(self, the_store):
        self.the_store = the_store

    def set_callback(self, callback):
        self.callback = callback

    def add_dependencies(self, dependents):
        if self.init_deps:
            raise RuntimeError("Dependencies already initialized!")
        if dependents is None:
            dependents = []
        self.dependents = dependents
        self.init_deps = True

    def add_physical_dependencies(self, all_props=None, physical_props=None):
        """Compute and store the direct physical dependencies based on either
        the set of all properties or the known physical properties"""
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

        if physical_props is not None:
            these_physical_props = set.intersection(these_deps, physical_props)
        else:
            if all_props is not None:
                these_physical_props = these_deps - set(all_props)
            else:
                raise RuntimeError("Need either all_props or physical_props to compute the correct physical dependencies!")
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
        self.init_physical_deps = True


    def add_implicit_dependencies(self, prop_map):
        if (not self.init_deps) or (not self.init_physical_deps):
            raise RuntimeError(
                "Dependencies and physical dependencies must be initialized before initializing implicit dependencies!"
            )

        if len(self.implicit_physical_props) == 0:
            prop_deps = set()
            for dependent_prop in self.props:
                dependent_context = prop_map[dependent_prop].context
                dependent_context.add_implicit_dependencies(prop_map)
                prop_deps |= set(dependent_context.physical_props)
                prop_deps |= set(dependent_context.implicit_physical_props)
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
        return tuple(values)

    def compute(self, parameters, physical_parameters, *args, **kwargs):
        values_getter = lambda : self.extract_values(parameters, physical_parameters)
        return self.callback(parameters, values_getter)

    def __call__(self, physical_parameters, *args, **kwargs):
        these_params = self.extract_params(physical_parameters)
        return self.compute(these_params, physical_parameters, *args, **kwargs)

def obtain_constants(root_name, dependents, f):
    args = [Parameter(str(d)) for d in dependents]
    root_node = entry.atomic_operation(*args)
    constants = dict()
    for node in toposort(root_node):
        if type(node) is Constant:
            constants[name] = node.value
    return constants

class gradient_information:
    def __init__(self, fbase, callback=None):
        self.fbase = fbase
        self.explored = False
        self.const_cached = False
        self.exploded = False
        self.root_node = None
        self.constants = dict()
        self.precomputed_information = None
        self.callback = callback

    def set_callback(self, callback):
        self.callback = callback

    # Basic numerical evaluation
    # Requires parameters to be ordered
    def eval_normal(self, parameters):
        return self.callback(*parameters)

    # Evaluation with gradient tracking
    # Accepts only parameter_wrappers
    # Requires parameter wrappers to be ordered
    def eval_normal_grad(self, parameter_wrappers):
        arg_names = self.fbase.arg_names
        nodes = [
            Node(name, [], value=pwrap)
            for name, pwrap in zip(arg_names, parameter_wrappers)
        ]
        res_node = self.callback(*nodes)
        return res_node.value

    def __call__(self, *args):
        new_args = []
        have_grad = False
        arg_names = self.fbase.arg_names
        for name, arg in zip(arg_names, args):
            if isinstance(arg, parameter_wrapper):
                have_grad = True
            else:
                arg = parameter_wrapper(name, arg)
            new_args.append(arg)
        if have_grad:
            return self.eval_normal_grad(new_args)
        else:
            return self.eval_normal(args)

class unpacker:
    def __init__(self, callback):
        self.callback = callback
    def __call__(self, key, value_getter):
        return self.callback(*value_getter())


class function_base:
    def __init__(self, name, arg_names, function):
        self.name = name
        self.arg_names = arg_names
        self.function = function
    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)


class function_wrapper:
    def __init__(self, name, arg_names, function):
        self.fbase = function_base(name, arg_names, function)
        self.arg_names = arg_names
        self.grad = gradient_information(self.fbase, callback=function)
        self.cache = function_cache(unpacker(self.grad), 1)
        self.context = entry_context(name, dependents=arg_names)
        self.context.set_callback(self.cache)

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

    def set_cache(self, cache):
        self.cache = cache

    def initialize_cache(self):
        self.cache

    def determine_context_callback(self,):
        cache_enabled = self.cache.enabled
        grad_enabled = True
        return "cache"

    def __call__(self, *args, **kwargs):
        return self.context(*args, **kwargs)


if __name__ == "__main__":

    def f(a, b, c, d):
        y = a + b
        z = c + d
        r = (y**2) * z
        return r

    f = function_wrapper(f, "f", ["a", "b", "c", "d"])


    a = parameter_wrapper('a', 1, grads=['g'], grad_values=[1])
    b = parameter_wrapper('b', 1, grads=['g'], grad_values=[1])
    c = parameter_wrapper('c', 1, grads=['h'], grad_values=[1])
    d = parameter_wrapper('d', 1, grads=['h'], grad_values=[1])

    res = f(a, b, c, d)
    print(res)
    print(res.value)
    print(res.grads)
    print(res.grad_values)
    res = f(a, 1.0, c, d)
    print(res)
    print(res.value)
    print(res.grads)
    print(res.grad_values)
    res = f(1.0, 1.0, 1.0, 1.0)
    print(res)
