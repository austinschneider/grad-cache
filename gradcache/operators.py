import numpy as np
import autodiff as ad

from parameter_wrapper import parameter_wrapper, sift_parameters

def evaluate_grad_operator(op_grad, parameter_wrappers, ngrads=None, names=None, final_indices=None):
    if ngrads is None or names is None or final_indices is None:
        ngrads, names, final_indices = sift_parameters(parameter_wrappers)

    args = [(x.value, np.array([[]]) if x.grad_values is None else x.grad_values) for x in parameter_wrappers]
    args += final_indices
    args += [ngrads]

    res, res_grad = op_grad(*args)
    return parameter_wrapper(None, res, grads=names, grad_values=res_grad)

class binary_operator:
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

class nary_operator:
    def __init__(self, name, op_base, op_grad, packed=False):
        self.name = name
        self.op_base = op_base
        self.op_grad = op_grad
        self.packed = packed

    def _eval_unpacked(self, params, ngrads=None, names=None, final_indices=None):
        pv = []
        pg = []
        b = []
        for param in params:
            if isinstance(param, parameter_wrapper):
                pvv, pgg = param.value, param.grad_values
            else:
                pvv, pgg = param, np.array([[]])
                b.append(pgg is not None)

        if np.any(b):
            res = evaluate_grad_operator(self.op_grad,
                    params,
                    ngrads=ngrads,
                    names=names,
                    final_indices=final_indices)
        else:
            res_value = self.op_base(*pv)
            res_grad = None
            res = parameter_wrapper(None, res_value, grads=names, grad_values=res_grad)
        return res

operators = {
        'plus': binary_operator('plus', ad.plus, ad.plus_10, ad.plus_01, ad.plus_grad),
        'minus': binary_operator('minus', ad.minus, ad.minus_10, ad.minus_01, ad.minus_grad),
        'mul': binary_operator('mul', ad.mul, ad.mul_10, ad.mul_01, ad.mul_grad),
        'div': binary_operator('div', ad.div, ad.div_10, ad.div_01, ad.div_grad),
        'pow': binary_operator('pow', ad.pow, ad.pow_10, ad.pow_01, ad.pow_grad),
        'inv': unary_operator('inv', ad.inv, ad.inv_grad),
        'lgamma': unary_operator('lgamma', ad.lgamma, ad.lgamma_grad),
        'log': unary_operator('log', ad.log, ad.log_grad),
        'log10': unary_operator('log10', ad.log10, ad.log10_grad),
        'log2': unary_operator('log2', ad.log2, ad.log2_grad),
        'sqrt': unary_operator('sqrt', ad.sqrt, ad.sqrt_grad),
        'sum': unary_operator('sum', ad.sum, ad.sum_grad),
        }


if __name__ == "__main__":
    p = operators['plus']

    def f(a, b, c, d):
        y = a + b
        z = c + d
        r = (y**2) * z
        return r

    def f(a, b, c, d):
        return operators['mul'].eval(operators['pow'].eval(operators['plus'].eval(a, b), 2), operators['plus'].eval(c, d))

    a = parameter_wrapper('a', 1, grads=['g'], grad_values=[1])
    b = parameter_wrapper('b', 1, grads=['g'], grad_values=[1])
    c = parameter_wrapper('c', 1, grads=['h'], grad_values=[1])
    d = parameter_wrapper('d', 1, grads=['h'], grad_values=[1])

    res = f(a, b, c, d)
    print(res.value)
    print(res.grads)
    print(res.grad_values)
