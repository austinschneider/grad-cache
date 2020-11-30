import os
import os.path
import collections
import numpy as np
from .node import Node, Constant, Parameter, name_nodes, toposort


# two types of arguments for accessing the store
# dict() --> parameters , dict --> grad
# dict() --> (parameter, grad)

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

def obtain_constants(root_name, dependents, f):
    args = [Parameter(str(d)) for d in dependents]
    root_node = entry.atomic_operation(*args)
    constants = dict()
    for node in toposort(root_node):
        if type(node) is Constant:
            constants[name] = node.value
    return constants

class store_entry:
    def __init__(self, name, dependents, atomic_operation):
        self.name = name
        if dependents is None:
            dependents = []
        self.dependents = dependents
        self.atomic_operation = atomic_operation
    def __call__(self, *args):
        return self.atomic_operation(*args)

class store_initialized_entry(store_entry):
    def __init__(self, uninitialized, the_store, physical_props, props):
        store_entry.__init__(self, uninitialized.name, uninitialized.dependents, uninitialized.atomic_operation)

        self.the_store = the_store
        self.implicit_physical_props = []
        self.physical_props = []
        self.props = []
        self.physical_props_indices = []
        self.props_indices = []
        for i,name in enumerate(self.dependents):
            if name in physical_props:
                self.physical_props.append(name)
                self.physical_props_indices.append(i)
            elif name in props:
                self.props.append(name)
                self.props_indices.append(i)
            else:
                raise ValueError("Not in props or physical_props:", name)

    def compute(self, parameters, physical_parameters, *args, **kwargs):
        values = [None for i in range(len(self.dependents))]
        for v,i in zip(parameters, self.physical_props_indices):
            values[i] = v
        for prop,i in zip(self.props, self.props_indices):
            values[i] = self.the_store.get_prop(prop, physical_parameters)
        return self.atomic_operation(*values)

    def __call__(self, physical_parameters, *args, **kwargs):
        these_params  = tuple([physical_parameters[k] for k in self.physical_props])
        these_params += tuple([physical_parameters[k] for k in self.implicit_physical_props])
        return self.compute(these_params, physical_parameters, *args, **kwargs)

class store_cached_entry(store_initialized_entry):
    def __init__(self, initialized, cache_size=None):
        store_initialized_entry.__init__(self, initialized, initialized.the_store, initialized.physical_props, initialized.props)

        if cache_size is None:
            cache_size = 1
        self.cache_size = cache_size

        self.cache = memodict(self.compute, self.cache_size)

    def __call__(self, physical_parameters, *args, **kwargs):
        these_params  = tuple([physical_parameters[k] for k in self.physical_props])
        these_params += tuple([physical_parameters[k] for k in self.implicit_physical_props])
        return self.cache(these_params, physical_parameters)

class const_cache:
    def __init__(self, root_name, dependents, atomic_operation, cache_size=None):
        self.root_name = root_name
        self.dependents = dependents
        self.atomic_operation

        if cache_size is None:
            cache_size = 1
        self.cache_size = cache_size

        self.cached = False
        self.cache = None

    def set_value(self, value):
        self.cache = value
        self.cached = True

    def compute(self, *args, **kwargs):
        constants = obtain_constants(self.root_name, self.dependents, self.atomic_operation)
        return constants

    def __call__(self, *args, **kwargs):
        if self.cached:
            return self.cache
        else:
            res = self.compute()
            if cache_size > 0:
                self.cache = res
                self.cached = True
            return res


class store_constants_cached_entry(store_initialized_entry):
    def __init__(self, name, cache):
        self.name = name
        self.implicit_physical_props = []
        self.physical_props = []
        self.props = []
        self.physical_props_indices = []
        self.props_indices = []
        self.dependents = []

        self.cache = cache

    def __call__(self, *args, **kwargs):
        return self.cache()[self.name]

class store_cached_grad_entry(store_initialized_entry):
    def __init__(self, initialized, cache_size=None):
        store_initialized_entry.__init__(self, initialized, initialized.the_store, initialized.physical_props, initialized.props)

        if cache_size is None:
            cache_size = 1
        self.cache_size = cache_size

        self.cache = memodict(self.compute, self.cache_size)

    def compute(self, parameters, physical_parameters, *args, **kwargs):
        values = [None for i in range(len(self.dependents))]
        for v,i in zip(parameters, self.physical_props_indices):
            values[i] = v
        for prop,i in zip(self.props, self.props_indices):
            values[i] = self.the_store.get_prop(prop, physical_parameters, *args, **kwargs)
        return self.atomic_operation(*values)

    def __call__(self, physical_parameters, *args, **kwargs):
        if "grads" in kwargs:
            grads = kwargs["grads"]

        these_params  = tuple([physical_parameters[k] for k in self.physical_props])
        these_params += tuple([physical_parameters[k] for k in self.implicit_physical_props])
        return self.cache(these_params, physical_parameters)


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
        return self.initialized_cache_props[name](physical_parameters, *args, **kwargs)

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
                node_entry = store_entry(name, dependents, atomic_operation, probe_func=False)
                props[node.name] = node_entry

            ##store_entry(node.name, dependents, atomic_operation, probe_func)


    def add_prop(self, name, dependents, atomic_operation, cache_size=None, probe_func=None):
        if probe_func is None:
            probe_func = self.default_probe_func
        prop = store_entry(name, dependents, atomic_operation)
        #if probe_func:
        #    self.expand_graph(prop)
        self.props[name] = prop
        self.cache_sizes[name] = cache_size

    def reset_cache(self, props):
        for prop in props:
            self.initialized_props[prop].clear()

    def initialize(self, keep_cache=False):
        # First see if we need to keep the old initialized caches around
        old_initialized_cache_props = self.initialized_cache_props
        if not keep_cache:
            del old_initialized_cache_props

        # Set up the location for the initialized entries
        self.initialized_props = dict()
        props = self.props.keys()

        # Collect all the dependents from across the entries
        dependents = set()
        for prop in props:
            deps = self.props[prop].dependents
            if deps is not None:
                dependents.update(deps)

        # Collect the names of all entries
        props = set(props)
        # See which of the dependencies are not functions registered with the store
        # These are our physical parameters
        physical_props = dependents - props

        # For each entry we need to get the physical properties that it depends on
        for prop in props:
            entry = self.props[prop]
            if entry.dependents is not None:
                these_deps = set(entry.dependents)
            else:
                these_deps = set()
            these_physical_props = these_deps - props
            these_props = these_deps - these_physical_props
            initialized_entry = store_initialized_entry(entry, self, these_physical_props, these_props)
            self.initialized_props[prop] = initialized_entry

        # Now we need another loop to make these into caches
        for prop in props:
            initialized_entry = self.initialized_props[prop]
            cache_entry = store_cached_entry(initialized_entry, cache_size=self.cache_sizes[prop])
            self.initialized_cache_props[prop] = cache_entry


        # There is no harm in initializing the caches in the lines above since they are empty
        # We can now replace them with the old copies if asked to
        if keep_cache:
            keys = old_initialized_cahe_props.keys()
            for k in keys:
                if k in self.initialized_cache_props:
                    self.initialized_cache_props[k].cache = old_initialized_cache_props[k].cache
            del old_initialized_cache_props

        # Compute the implicit physical dependencies recursively
        def add_implicit_dependencies(prop):
            initialized_cache_entry = self.initialized_cache_props[prop]
            if len(initialized_cache_entry.implicit_physical_props) == 0:
                prop_deps = set()
                for dprop in initialized_cache_entry.props:
                    add_implicit_dependencies(dprop)
                    prop_deps |= set(self.initialized_cache_props[dprop].physical_props)
                    prop_deps |= set(self.initialized_cache_props[dprop].implicit_physical_props)
                prop_deps -= set(initialized_cache_entry.physical_props)
                initialized_cache_entry.implicit_physical_props = list(prop_deps)

        # Ensure the implicit dependencies exist for all entries
        for prop in props:
            add_implicit_dependencies(prop)

    def __getitem__(self, args):
        prop_name, parameters = args
        return self.get_prop(prop_name, parameters)

class store_view:
    def __init__(self, the_store, parameters):
        self.the_store = the_store
        self.parameters = parameters
    def __getitem__(self, prop_name):
        return self.the_store.get_prop(prop_name, self.parameters)

