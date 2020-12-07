import numpy as np

try:
    from .node import Node
    from .parameter_wrapper import parameter_wrapper, sift_parameters
    from .cache import function_cache
    from .context import function_context, unpacker
    from .gradient import function_gradient
except:
    from node import Node
    from parameter_wrapper import parameter_wrapper, sift_parameters
    from cache import function_cache
    from context import function_context, unpacker
    from gradient import function_gradient


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
        self.grad = function_gradient(self.fbase, callback=function)
        self.cache = function_cache(unpacker(self.grad), 1)
        self.context = function_context(name, dependents=arg_names)
        self.context.set_callback(self.cache)

    name = property(lambda self: self.fbase.name)
    arg_names = property(lambda self: self.fbase.arg_names)
    function = property(lambda self: self.fbase.function)

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
