import os
import os.path
import collections
import numpy as np
from .node import Node, Constant

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

class store_entry:
    def __init__(self, name, dependents, atomic_operation, probe_func):
        self.name = name
        if dependents is None:
            dependents = []
        self.dependents = dependents
        self.atomic_operation = atomic_operation
        self.node = None

        # Optionally construct a computation graph of the function internals
        if probe_func:
            args = [Constant(str(d)) for d in dependents]
            self.node = self.atomic_operation(*args)
    def __call__(self, *args):
        return self.atomic_operation(*args)

class store_initialized_entry(store_entry):
    def __init__(self, uninitialized, the_store, physical_props, props, cache_size=None):
        store_entry.__init__(self, uninitialized.name, uninitialized.dependents, uninitialized.atomic_operation, probe_func=False)

        self.the_store = the_store

        if cache_size is None:
            cache_size = 1
        self.cache_size = cache_size

        self.implicit_physical_props = []
        self.physical_props = []
        self.props = []
        self.physical_props_indices = []
        self.props_indices = []
        self.is_physical = []
        for i,name in enumerate(self.dependents):
            if name in physical_props:
                self.is_physical.append(True)
                self.physical_props.append(name)
                self.physical_props_indices.append(i)
            elif name in props:
                self.is_physical.append(False)
                self.props.append(name)
                self.props_indices.append(i)
            else:
                raise ValueError("Not in props or physical_props:", name)

        self.cache = memodict(self.compute, self.cache_size)

    def compute(self, parameters, physical_parameters):
        values = [None for i in range(len(self.dependents))]
        for v,i in zip(parameters, self.physical_props_indices):
            values[i] = v
        for prop,i in zip(self.props, self.props_indices):
            values[i] = self.the_store.get_prop(prop, physical_parameters)
        return self.atomic_operation(*values)

    def __call__(self, physical_parameters):
        these_params  = tuple([physical_parameters[k] for k in self.physical_props])
        these_params += tuple([physical_parameters[k] for k in self.implicit_physical_props])
        return self.cache(these_params, physical_parameters)

class store:
    def __init__(self, default_cache_size=1, default_probe_func=True):
        self.default_cache_size = default_cache_size
        self.props = dict()
        self.cache_sizes = dict()
        self.initialized_props = dict()
        self.default_probe_func = default_probe_func

    def get_prop(self, name, physical_parameters=None):
        if physical_parameters is None:
            physical_parameters = dict()
        return self.initialized_props[name](physical_parameters)

    def add_prop(self, name, dependents, atomic_operation, cache_size=None, probe_func=None):
        if probe_func is None:
            probe_func = self.default_probe_func
        prop = store_entry(name, dependents, atomic_operation, probe_func)
        self.props[name] = prop
        self.cache_sizes[name] = cache_size

    def reset_cache(self, props):
        for prop in props:
            self.initialized_props[prop].clear()

    def initialize(self, keep_cache=False):
        # First see if we need to keep the old initialized caches around
        old_initialized_props = self.initialized_props
        if not keep_cache:
            del old_initialized_props

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

        # For each entry we need to get the physical properties that is depends on
        # and initialize the cache
        for prop in props:
            entry = self.props[prop]
            if entry.dependents is not None:
                these_deps = set(entry.dependents)
            else:
                these_deps = set()
            these_physical_props = these_deps - props
            these_props = these_deps - these_physical_props
            initialized_entry = store_initialized_entry(entry, self, these_physical_props, these_props, cache_size=self.cache_sizes[prop])
            self.initialized_props[prop] = initialized_entry


        # There is no harm in initializing the caches in the lines above since they are empty
        # We can now replace them with the old copies if asked to
        if keep_cache:
            keys = old_initialized_props.keys()
            for k in keys:
                if k in self.initialized_props:
                    self.initialized_props[k].cache = old_initialized_props[k].cache
            del old_initialized_props

        # Compute the implicit physical dependencies recursively
        def add_implicit_dependencies(prop):
            initialized_entry = self.initialized_props[prop]
            if len(initialized_entry.implicit_physical_props) == 0:
                prop_deps = set()
                for dprop in initialized_entry.props:
                    add_implicit_dependencies(dprop)
                    prop_deps |= set(self.initialized_props[dprop].physical_props)
                    prop_deps |= set(self.initialized_props[dprop].implicit_physical_props)
                prop_deps -= set(initialized_entry.physical_props)
                initialized_entry.implicit_physical_props = list(prop_deps)

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

