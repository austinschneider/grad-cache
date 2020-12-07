import numpy as np
import collections
from operator import itemgetter


class parameter_wrapper(tuple):
    __slots__ = []
    def __new__(cls, name, value, grads=None, grad_values=None):
        if grads is not None:
            grads = tuple(grads)
            if grad_values is None:
                grad_values = tuple((None for _ in range(len(grads))))
            else:
                grad_values = tuple(grad_values)
        else:
            grads = None
            grad_values = None
        return tuple.__new__(cls, (name, value, grads, grad_values))

    name = property(itemgetter(0))
    value = property(itemgetter(1))
    grads = property(itemgetter(2))
    grad_values = property(itemgetter(3))

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
