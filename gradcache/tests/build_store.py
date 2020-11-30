# -*- coding: utf-8 -*-
import numpy as np
from context import gradcache
import unittest


Node = gradcache.Node
Constant = gradcache.Constant
Parameter = gradcache.Parameter
store = gradcache.store


class BasicTest(unittest.TestCase):
    """Basic test cases."""

    def test_this(self):
        def a(g):
            print("a")
            return 1.0 * g

        def b(g):
            print("b")
            return 1.0 * g

        def c(h):
            print("c")
            return 1.0 * h

        def d(h):
            print("d")
            return 1.0 * h

        def f(a, b, c, d):
            y = a + b
            z = c + d
            r = y * z
            return r

        the_store = store(default_cache_size=1, default_probe_func=True)

        the_store.add_prop("f", ["a", "b", "c", "d"], f)
        the_store.add_prop("a", ["g"], a)
        the_store.add_prop("b", ["g"], b)
        the_store.add_prop("c", ["h"], c)
        the_store.add_prop("d", ["h"], d)
        the_store.initialize()

        params = {"g": 1.0, "h": 2.0}

        print("f:", the_store["f", params])
        print("f:", the_store["f", params])

        def get_name(node):
            c_str = ", ".join(
                [get_name(c) if type(c) is Node else str(c) for c in node.children]
            )
            if node.op is None:
                return c_str
            else:
                return str(node.op) + "(" + c_str + ")"

        return True


if __name__ == "__main__":
    unittest.main()
