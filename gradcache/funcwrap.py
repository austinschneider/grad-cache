import numpy as np

from parameter_wrapper import parameter_wrapper, sift_parameters



class function_wrapper:
    def __init__(self, function, arg_names):
        self.function = function
        self.arg_names = arg_names

    # basic numerical evaluation
    def eval(self, parameters):
        args = [parameters[k] for k in self.arg_names]
        return self.function(*args)

    # 
    def eval_grad(self, parameter_wrappers):
        n_grads, final_indices = sift_parameters(parameter_wrappers)

        pass

