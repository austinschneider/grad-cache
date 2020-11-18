import numpy as np

from parameter_wrapper import parameter_wrapper, sift_parameters

def evaluate_grad_operator(op_grad, parameter_wrappers, ngrads=None, names=None, final_indices=None):
    if n_grads is None or names is None or final_indices is None:
        n_grads, names, final_indices = sift_parameters(parameter_wrappers)

    args = [(x.value, x.grads) for x in parameter_wrappers]
    args += final_indices

    res, res_grad = p_grad(*args)
    return parameter_wrapper(None, res, grads=names, grad_values=res_grad)

class boolean_operator:
    def __init__(self, name, op_base, op_10, op_01, op_grad):
        self.name = name
        self.op_base = op_base
        self.op_10 = op_10
        self.op_01 = op_01
        self.op_grad = op_grad
        self.ops = [self.op_base, self.op_01, self.op_10, self.op_grad]

    def eval(self, param0, param1, ngrads=None, names=None, final_indices=None):
        if isinstance(param0, parameter_wrapper):
            p0v, p0g = param0.value, param0.grad_values
        else:
            p0v, p0g = param0, None
        if isinstance(param1, parameter_wrapper):
            p1v, p1g = param1.value, param1.grad_values
        else:
            p1v, p1g = param1, None

        b0 = p0g is not None
        b1 = p1g is not None
        b = b0 << 1 | b1
        op = self.ops[b]
        if b == 0:
            res_value = op(p0v, p1v)
            res_grad = None
            names = None
        elif b == 1:
            res_value, res_grad = op(p0v, (p1v, p1g))
            names = param1.grads
        elif b == 2:
            res_value, res_grad = op((p0v, p0g), p1v)
            names = param0.grads
        elif b == 3:
            res = evaluate_grad_operator(op, [param0, param1], ngrads=ngrads, names=names, final_indices=final_indices)
            return res

        return parameter_wrapper(None, res_value, grads=names, grad_values=res_grad)

class unary_operator:
    def __init__(self, name, op_base, op_grad):
        self.name = name
        self.op_base = op_base
        self.op_grad = op_grad
    
    def eval(self, param0, ngrads=None, names=None, final_indices=None):
        if isinstance(param0, parameter_wrapper):
            p0v, p0g = param0.value, param0.grad_values
        else:
            p0v, p0g = param0, None

        if p0g is None:
            res_value = self.op_base(p0v)
            res_grad = None
            names = None
        else:
            res_value, res_grad = self.op_grad((p0v, p0g))
            names = param0.grads

        return parameter_wrapper(None, res_value, grads=names, grad_values=res_grad)



