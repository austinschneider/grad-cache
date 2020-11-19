import numpy as np
import types
from functools import wraps
import gradcache.autodiff as ad


def register_op(cls, method, token, n, rev):
    if n == 2:
        op = str(token)
        if rev:
            def inner(self, other):
                otype = type(other)
                if otype is not Node:
                    other = Constant(other)
                return Node(op, [other, self])
        else:
            def inner(self, other):
                otype = type(other)
                if otype is not Node:
                    other = Constant(other)
                return Node(op, [self, other])
    elif n == 1:
        op = str(token)
        def inner(self):
            return Node(op, [self])

    setattr(cls, method, inner)


class Node:
    """A node in a computation graph."""
    def __init__(self, op, children):
        self.children = children
        self.op = op
        self.name = None
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
        ]

for op in operators:
    Node.register_op(*op)

def log(node):
    op = 'log'
    return Node(op, [node])

def log10(node):
    op = 'log10'
    return Node(op, [node])

def log2(node):
    op = 'log2'
    return Node(op, [node])

def sqrt(node):
    op = 'sqrt'
    return Node(op, [node])

def lgamma(node):
    op = 'lgamma'
    return Node(op, [node])

def log1p(node):
    op = 'log1p'
    return Node(op, [node])


"""
    def __add__(self, other):
        op = 'plus'
        return Node(op, [self, other])

    def __radd__(self, other):
        op = 'plus'
        return Node(op, [other, self])

    def __sub__(self, other):
        op = 'minus'
        return Node(op, [self, other])

    def __rsub__(self, other):
        op = 'minus'
        return Node(op, [other, self])

    def __mul__(self, other):
        op = 'mul'
        return Node(op, [self, other])

    def __rmul__(self, other):
        op = 'mul'
        return Node(op, [other, self])

    def __div__(self, other):
        op = 'div'
        return Node(op, [self, other])

    def __rdiv__(self, other):
        op = 'div'
        return Node(op, [other, self])

    def __pow__(self, other):
        op = 'pow'
        return Node(op, [self, other])

    def __rpow__(self, other):
        op = 'pow'
        return Node(op, [other, self])

    def __invert__(self):
        op = 'inv'
        return Node(op, [self])
"""
