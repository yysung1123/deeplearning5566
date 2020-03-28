from typing import Optional
from . import ops
from dl56 import Tensor

def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor]=None):
    r"""
    y = x * W^T + b
    x = (batch, input_size)
    W = (output_size, input_size)
    b = (output_size)
    """
    output = input.matmul(weight.t())
    if bias is not None:
        output = output + bias
    return output

def relu(input: Tensor):
    r"""
    y = relu(x)
    x = (batch, ...)
    """
    output = ops.Relu.apply(input)
    return output

def mse_loss(input: Tensor, target: Tensor):
    if not (input.data.shape == target.data.shape):
        raise Exception("MSELoss Error")
    res = (input - target) ** 2
    return ops.Mean.apply(res)

def log_softmax(input: Tensor):
    return ops.LogSoftmax.apply(input)

def nll_loss(input: Tensor, target: Tensor):
    if not (input.data.shape == target.data.shape):
        raise Exception("NLLLoss Error")
    return -ops.Mean.apply(input * target)

def cross_entropy(input: Tensor, target: Tensor):
    if not (input.data.shape == target.data.shape):
        raise Exception("CrossEntropy Error")
    return nll_loss(log_softmax(input), target)