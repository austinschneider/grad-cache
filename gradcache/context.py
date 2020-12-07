import numpy as np

class function_context:
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

class unpacker:
    def __init__(self, callback):
        self.callback = callback
    def __call__(self, key, value_getter):
        return self.callback(*value_getter())

