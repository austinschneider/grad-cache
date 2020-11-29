import numpy as np

from .parameter_wrapper import parameter_wrapper, sift_parameters

from .node import Node

class memodict_(collections.OrderedDict):
    def __init__(self, f, maxsize=1):
        collections.OrderedDict.__init__(self)
        self.f = f
        self.maxsize = maxsize
    def __getitem__(self, key, extra=None):
        if super().__contains__(key):
            return super().__getitem__(key)
        if len(self) == self.maxsize:
            self.popitem(last=False)
        ret = self.f(key, extra)
        super().__setitem__(key, ret)
        return ret
    def __call__(self, key, extra):
        return self.__getitem__(key, extra)

def memodict(f, maxsize=1):
    """ Memoization decorator for a function taking a single argument """
    m = memodict_(f, maxsize)
    return m

class memodict_(collections.OrderedDict):
    def __init__(self, f, maxsize=1, enabled=True, sample_time=True, sample_mem=True, track_time=False, track_mem=False):
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
        time_sample = (not len(self.time_samples) and self.sample_time) or self.track_time
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

class function_wrapper:
    def __init__(self, function, arg_names):
        self.function = function
        self.arg_names = arg_names
        self.cache = memodict(f, 1)

    def get_cache_state(self):
        pass

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

    # basic numerical evaluation
    def eval(self, parameters):
        args = [parameters[k] for k in self.arg_names]
        return self.function(*args)

    # Evaluation with gradient tracking
    # Accepts only parameter_wrappers
    def eval_grad(self, parameter_wrappers):
        nodes = [Node(name, [], value=pwrap) for name,pwrap in zip(self.arg_names,parameter_wrappers)]
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

