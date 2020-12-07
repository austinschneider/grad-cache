import os
import os.path
import collections
import numpy as np
try:
    from .node import Node, Constant, Parameter, name_nodes, toposort
    from .wrapper import function_wrapper
except:
    from node import Node, Constant, Parameter, name_nodes, toposort
    from wrapper import function_wrapper

class store:
    def __init__(self, default_cache_size=1, default_probe_func=True):
        self.default_cache_size = default_cache_size
        self.props = dict()
        self.cache_sizes = dict()
        self.initialized_props = dict()
        self.initialized_cache_props = dict()
        self.default_probe_func = default_probe_func

    def get_prop(self, name, physical_parameters=None, *args, **kwargs):
        if physical_parameters is None:
            physical_parameters = dict()
        return self.props[name](physical_parameters, *args, **kwargs)

    def expand_graph(self, entry):
        # Get the root node
        # This represents the return value of the function

        f = entry.atomic_operation
        root_name = entry.name
        root_node, constants = name_nodes(root_name, dependents, f)

        constant_cache = const_cache(root_name, dependents, f, cache_size=None)
        constant_cache.set_value(constants)

        entries = []
        entries.append(entry)

        constant_entries = []

        for node in toposort(root_node):
            name = node.name
            dependents = [c.name for c in node.children]
            # Now we need to implement a special type of entry and some extra functionality
            # The trick here is that there is not exactly a single atomic operation, but rather a series of them
            # There will be one operator per possible gradient query, i.e the scalar only case, the single variable derivative case, the 2d gradient, the 3d gradient, etc.
            # We probably need a class to represent a particular operator (ex. addition) which then covers these cases
            # We also will need some special caching functionality
            # The function we call now depends on what kind of gradient query we make

            if type(node) is Constant:
                # Define the function to obtain constants
                const_entry = store_constants_cached_entry(node.name, constant_cache)
                props[node.name] = const_entry

            elif type(node) is Node:
                atomic_operation = make_atomic_op()
                node_entry = store_entry(
                    name, dependents, atomic_operation, probe_func=False
                )
                props[node.name] = node_entry

            ##store_entry(node.name, dependents, atomic_operation, probe_func)

    def add_prop(
        self, name, dependents, atomic_operation, cache_size=None, probe_func=None
    ):
        if probe_func is None:
            probe_func = self.default_probe_func
        prop = function_wrapper(name, dependents, atomic_operation)
        # if probe_func:
        #    self.expand_graph(prop)
        self.props[name] = prop
        self.cache_sizes[name] = cache_size

    def initialize_function_contexts(self):
        prop_dict = self.props
        props = prop_dict.keys()

        # Collect all the dependents from across the entries
        dependents = set()
        for prop in props:
            func_wrap = prop_dict[prop]
            deps = func_wrap.arg_names
            func_wrap.context.set_store(self)
            if deps is not None:
                dependents.update(deps)

        # Collect the names of all entries
        props = set(props)
        # See which of the dependencies are not functions registered with the store
        # These are our physical parameters
        physical_props = dependents - props

        # For each entry we need to set the physical properties that it depends on
        for prop in props:
            prop_dict[prop].context.add_physical_dependencies(physical_props=physical_props)

    def add_implicit_physcial_dependencies(self):
        props = self.props.keys()
        for prop in props:
            self.props[prop].context.add_implicit_dependencies(self.props)

    def extract_caches(self):
        prop_dict = self.props
        props = prop_dict.keys()

        caches = dict()
        for prop in props:
            caches[prop] = prop_dict[prop].cache
        return caches

    def set_caches(self, caches):
        prop_dict = self.props
        props = prop_dict.keys()

        for prop in props:
            prop_dict[prop].set_cache(caches[prop])

    def reset_caches(self, props):
        prop_dict = self.props
        props = prop_dict.keys()

        for prop in props:
            prop_dict[prop].clear()

    def initialize_caches(self):
        prop_dict = self.props
        props = prop_dict.keys()

        for prop in props:
            prop_dict[prop].initialize_cache()

    def initialize(self, keep_cache=False):
        if keep_cache:
            old_caches = self.extract_caches()

        self.initialize_function_contexts()
        self.add_implicit_physcial_dependencies()
        self.initialize_caches()

        if keep_cache:
            self.set_caches(old_caches)

    def __getitem__(self, args):
        prop_name, parameters = args
        return self.get_prop(prop_name, parameters)


class store_view:
    def __init__(self, the_store, parameters):
        self.the_store = the_store
        self.parameters = parameters

    def __getitem__(self, prop_name):
        return self.the_store.get_prop(prop_name, self.parameters)

if __name__ == "__main__":
    try:
        from .parameter_wrapper import parameter_wrapper, sift_parameters
    except:
        from parameter_wrapper import parameter_wrapper, sift_parameters

    def a(g):
        print("a")
        return 1.0 * g

    def b(g):
        print("b")
        return 1.0 * g

    def c(h):
        print("c")
        return 1.0 * h

    def d(h):
        print("d")
        return 1.0 * h

    def f(a, b, c, d):
        print("f")
        y = a + b
        z = c + d
        r = y * z
        return r

    the_store = store(default_cache_size=1, default_probe_func=True)

    the_store.add_prop("f", ["a", "b", "c", "d"], f)
    the_store.add_prop("a", ["g"], a)
    the_store.add_prop("b", ["g"], b)
    the_store.add_prop("c", ["h"], c)
    the_store.add_prop("d", ["h"], d)
    the_store.initialize()

    params = {"g": 1.0, "h": 2.0}

    print("f:", the_store["f", params])
    print("f:", the_store["f", params])

    params = {
            "g": parameter_wrapper("g", 1.0, grads=["g"], grad_values=[1]),
            "h": parameter_wrapper("h", 2.0, grads=["h"], grad_values=[1]),
            }

    res = the_store["f", params]
    print(res)
    print(res.value)
    print(res.grads)
    print(res.grad_values)
    res = the_store["f", params]
    print(res)
    print(res.value)
    print(res.grads)
    print(res.grad_values)
