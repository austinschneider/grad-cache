import numpy as np
import types
from functools import wraps
#import gradcache.autodiff as ad
import autodiff as ad
from operators import operators as ops


def get_value(x):
    if isinstance(x, Node):
        return x.value
    else:
        return x
    return x.value if isinstance(x, Node) else x

def build_op(token, n, rev):
    if n == 2:
        op = str(token)
        if rev:
            def inner(self, other):
                if self.evaluate:
                    sval = get_value(self)
                    oval = get_value(other)
                    res = ops[op].eval(oval, sval)
                    return Node(op, [], value=res)
                else:
                    if not isinstance(other, Node):
                        other = Constant(other)
                    return Node(op, [other, self])
        else:
            def inner(self, other):
                if self.evaluate:
                    sval = get_value(self)
                    oval = get_value(other)
                    res = ops[op].eval(sval, oval)
                    return Node(op, [], value=res)
                else:
                    if not isinstance(otype, Node):
                        other = Constant(other)
                    return Node(op, [self, other])
    elif n == 1:
        op = str(token)
        def inner(self):
            if self.evaluate:
                sval = get_value(self)
                res = ops[op].eval(sval)
                return Node(op, [], value=res)
            else:
                return Node(op, [self])
    return inner


def register_op(cls, method, token, n, rev):
    op = build_op(token, n, rev)
    setattr(cls, method, op)


class Node:
    """A node in a computation graph."""
    def __init__(self, op, children, value=None):
        self.children = children
        self.op = op
        self.name = None
        if value is None:
            self.evaluate = False
            self.value = None
        else:
            self.evaluate = True
            self.value = value
    @classmethod
    def register_op(cls, *args):
        register_op(cls, *args)

    def __repr__(self):
        c_str = ", ".join([c.__repr__() if type(c) is Node else str(c) for c in self.children])
        if self.op is None:
            return c_str
        else:
            return str(self.op) + "(" + c_str + ")"

class Constant(Node):
    """A constant node"""
    def __init__(self, value):
        Node.__init__(self, None, [])
        self.value = value
    def __repr__(self):
        return str(self.value)

class Parameter(Node):
    """A variable parameter node"""
    def __init(self, name):
        Node.__init__(self, None, [])
        self.name = name
    def __repr__(self):
        return str(self.name)

operators = [
        ("__add__", "plus", 2, False),
        ("__radd__", "plus", 2, True),
        ("__sub__", "minus", 2, False),
        ("__rsub__", "minus", 2, True),
        ("__mul__", "mul", 2, False),
        ("__rmul__", "mul", 2, True),
        ("__div__", "div", 2, False),
        ("__rdiv__", "div", 2, True),
        ("__pow__", "pow", 2, False),
        ("__rpow__", "pow", 2, True),
        ("__invert__", "inv", 1, False),
        ("log", "log", 1, False),
        ("log10", "log10", 1, False),
        ("log2", "log2", 1, False),
        ("sqrt", "sqrt", 1, False),
        ("lgamma", "lgamma", 1, False),
        ("log1p", "log1p", 1, False),
        ]

for op in operators:
    Node.register_op(*op)

def set_op(obj, method, token, n, rev):
    op = build_op(token, n, rev)
    obj[method] = op

methods = [
        "log",
        "log10",
        "log2",
        "sqrt",
        "lgamma",
        "log1p",
        ]

for method in methods:
    set_op(globals(), method, method, 1, False)


if __name__ == "__main__":

    def f(a, b, c, d):
        y = a + b
        z = c + d
        r = (y**2) * z
        return r

    from parameter_wrapper import parameter_wrapper, sift_parameters

    a = parameter_wrapper('a', 1, grads=['g'], grad_values=[1])
    b = parameter_wrapper('b', 1, grads=['g'], grad_values=[1])
    c = parameter_wrapper('c', 1, grads=['h'], grad_values=[1])
    d = parameter_wrapper('d', 1, grads=['h'], grad_values=[1])

    na = Node("a", [], value=a)
    nb = Node("b", [], value=b)
    nc = Node("c", [], value=c)
    nd = Node("d", [], value=d)

    res = f(na, nb, nc, nd)
    print(res.value)
    print(res.value.value)
    print(res.value.grads)
    print(res.value.grad_values)

