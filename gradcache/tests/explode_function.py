# -*- coding: utf-8 -*-
import numpy as np
from context import gradcache
import unittest


Node = gradcache.Node
Constant = gradcache.Constant
Parameter = gradcache.Parameter

class BasicTest(unittest.TestCase):
    """Basic test cases."""

    def test_this(self):
        def a(g):
            return 1.0*g
        def b(g):
            return 1.0*g
        def c(h):
            return 1.0*h
        def d(h):
            return 1.0*h

        def f(a, b, c, d):
            y = a+b
            z = c+d
            r = y*z
            return r

        g = Parameter("g")
        h = Parameter("h")

        res = f(a(g), b(g), c(h), d(h))

        print(res)

        def get_name(node):
            c_str = ", ".join([get_name(c) if type(c) is Node else str(c) for c in node.children])
            if node.op is None:
                return c_str
            else:
                return str(node.op) + "(" + c_str + ")"

        return True


if __name__ == "__main__":
    unittest.main()
