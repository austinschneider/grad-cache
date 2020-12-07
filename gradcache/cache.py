import numpy as np
import collections
import time

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

