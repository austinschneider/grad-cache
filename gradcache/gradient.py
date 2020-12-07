
try:
    from .parameter_wrapper import parameter_wrapper, sift_parameters
    from .node import Node
except:
    from parameter_wrapper import parameter_wrapper, sift_parameters
    from node import Node

class function_gradient:
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

def obtain_constants(root_name, dependents, f):
    args = [Parameter(str(d)) for d in dependents]
    root_node = entry.atomic_operation(*args)
    constants = dict()
    for node in toposort(root_node):
        if type(node) is Constant:
            constants[name] = node.value
    return constants

