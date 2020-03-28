from . import functional as F

import dl56
import numpy as np

class Parameter(dl56.Tensor):
    def __init__(self, data):
        super().__init__(data=data, requires_grad=True)

class Module:
    def parameters(self):
        param_list = []
        for name, param in self.__dict__.items():
            if isinstance(param, Parameter):
                param_list.append(param)
            if isinstance(param, Module):
                param_list.extend(param.parameters())
        return param_list

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        self.weight = self.kaiming_normal(in_features, (out_features, in_features))
        self.bias = self.kaiming_normal(in_features, (out_features,))
        super().__init__()

    def kaiming_normal(self, in_features, shape):
        return Parameter(data=np.random.randn(*shape) / np.sqrt(in_features / 2))

    def __call__(self, input):
        return F.linear(input, self.weight, self.bias)