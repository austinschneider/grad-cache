import numpy as np

from parameter_wrapper import parameter_wrapper, sift_parameters

from node import Node


class function_wrapper:
    def __init__(self, function, arg_names):
        self.function = function
        self.arg_names = arg_names

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

if __name__ == "__main__":

    def f(a, b, c, d):
        y = a + b
        z = c + d
        r = (y**2) * z
        return r

    f = function_wrapper(f, ["a", "b", "c", "d"])


    a = parameter_wrapper('a', 1, grads=['g'], grad_values=[1])
    b = parameter_wrapper('b', 1, grads=['g'], grad_values=[1])
    c = parameter_wrapper('c', 1, grads=['h'], grad_values=[1])
    d = parameter_wrapper('d', 1, grads=['h'], grad_values=[1])

    res = f(a, b, c, d)
    res = f(a, 1.0, c, d)
    print(res)
    print(res.value)
    print(res.grads)
    print(res.grad_values)
