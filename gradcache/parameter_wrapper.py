import numpy as np
import collections


class parameter_wrapper:
    def __init__(self, name, value, grads=None, grad_values=None):
        # Just a string
        self.name = name

        # A numeric value
        # from node import Node
        # assert(not isinstance(value, parameter_wrapper))
        # assert(not isinstance(value, Node))
        self.value = value

        # names of parameters for which we should track the gradient
        self.grads = grads

        self.grad_values = None

        if self.grads is not None:
            if grad_values is None:
                self.grad_values = [None for _ in range(len(self.grads))]
            else:
                self.grad_values = grad_values

    def primitive(self):
        if self.grad_values is None:
            return self.value
        else:
            return self.value, self.grad_values


### Helper functions


def sift_parameters(parameters):

    all_grads = collections.OrderedDict()

    grads_counter = 0

    final_indices = []

    for i, p in enumerate(parameters):
        if not isinstance(p, parameter_wrapper):
            print("sift_parameters only takes objects of type parameter_wrapper!")
            raise

        name = p.name
        grads = p.grads

        if grads is None:
            final_indices.append(np.array([]).astype(int))
            continue

        grad_final_indices = []
        for j, g in enumerate(grads):
            if g not in all_grads:
                all_grads[g] = grads_counter
                grads_counter += 1
            grad_final_indices.append(all_grads[g])
        final_indices.append(np.array(grad_final_indices))

    return grads_counter, list(all_grads.keys()), final_indices
