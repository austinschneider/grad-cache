import numpy as np
import queue

from .node import Node, Constant, Parameter, name_nodes, toposort, get_value
from .operators import operators as ops

def evaluate_graph(root_node, args):
    q = queue.Queue()
    q.put(root_node)
    while not q.empty():
        node = q.get()
        if isinstance(node, Constant):
            continue
        elif isinstance(node, Parameter):
            node.value = args[node.name]
        elif node.value is None:
            children = node.children
            have_values = True
            for child in children:
                if child.value is None:
                    have_values = False
                    q.put(child)
            if have_values:
                node.value = ops[node.op].eval(*[get_value(c) for c in children])
                for child in children:
                    child.value = None
                    child.evaluate = False
            else:
                q.put(node)
        else:
            print("value", node.value)
    return root_node.value


if __name__ == "__main__":

    def f(a, b, c, d):
        y = a + b
        z = c + d
        r = y * z
        return r

    from parameter_wrapper import parameter_wrapper, sift_parameters

    a = parameter_wrapper('a', 1, grads=['g', 'h'], grad_values=[1, 2])
    b = parameter_wrapper('b', 1, grads=['g'], grad_values=[1])
    c = parameter_wrapper('c', 1, grads=['h'], grad_values=[1])
    d = parameter_wrapper('d', 1, grads=['h'], grad_values=[1])

    na = Parameter("a", value=a)
    nb = Parameter("b", value=b)
    nc = Parameter("c", value=c)
    nd = Parameter("d", value=d)

    res = f(na, nb, nc, nd)
    print(res)
    print(res.value)
    print(res.value.value)
    print(res.value.grads)
    print(res.value.grad_values)
    print()

    res.reset()
    root_node = res

    print()
    v = evaluate_graph(root_node, {"a": 1, "b": 1, "c": 1, "d": 1})
    print(v)
    print(v.value)
    print()

    g = parameter_wrapper('g', 1, grads=['g'], grad_values=[1])
    h = parameter_wrapper('h', 1, grads=['h'], grad_values=[1])

    root_node.reset()

    print()
    res = evaluate_graph(root_node, {"a": a, "b": b, "c": c, "d": d})
    print(res)
    print(res.value)
    print(res.grads)
    print(res.grad_values)



