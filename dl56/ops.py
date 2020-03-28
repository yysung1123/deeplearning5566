from .util import maybe_unexpand
from . import tensor
from dl56 import autograd

import numpy as np

class Add(autograd.Function):
    # a + b
    @staticmethod
    def forward(ctx, a, b):
        # print(a.data.shape, b.data.shape)
        ctx.save_for_backward(a, b)
        return tensor.Tensor(a.data + b.data)

    # y = a + b
    # dy/da = 1 dy/db = 1
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return maybe_unexpand(grad, a.data.shape), maybe_unexpand(grad, b.data.shape)

class Sub(autograd.Function):
    # a - b
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return tensor.Tensor(a.data - b.data)

    # y = a - b
    # dy/da = 1, dy/db = -1
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return maybe_unexpand(grad, a.data.shape), maybe_unexpand(-grad, a.data.shape)

class Mul(autograd.Function):
    # element wise multiplication
    # a * b
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return tensor.Tensor(a.data * b.data)

    # y = a * b
    # dy/da = b, dy/db = a
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return grad * b.data, grad * a.data

class Neg(autograd.Function):
    # element wise negative
    # -a
    @staticmethod
    def forward(ctx, a):
        return tensor.Tensor(-a.data)

    # y = -a
    # dy/da = -1
    @staticmethod
    def backward(ctx, grad):
        return -grad

class Transpose(autograd.Function):
    @staticmethod
    def forward(ctx, a):
        return tensor.Tensor(a.data.T)

    @staticmethod
    def backward(ctx, grad):
        return grad.T

class Matmul(autograd.Function):
    # a @ b
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return tensor.Tensor(a.data @ b.data)

    # C = A * B
    # dz/dA = dz/dC * B^T
    # dz/dB = A^T * dz/dC
    @staticmethod
    def backward(ctx, grad):
        a, b = ctx.saved_tensors
        return grad @ b.data.T, a.data.T @ grad

class Relu(autograd.Function):
    # relu(a)
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return tensor.Tensor(np.fmax(a.data, 0))

    @staticmethod
    def backward(ctx, grad):
        a, = ctx.saved_tensors
        return grad * (a.data > 0)

class Sum(autograd.Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return tensor.Tensor(np.array([a.data.sum()]))

    @staticmethod
    def backward(ctx, grad: int):
        a, = ctx.saved_tensors
        return np.full(a.data.shape, grad)

class Mean(autograd.Function):
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        return tensor.Tensor(np.array([a.data.mean()]))

    @staticmethod
    def backward(ctx, grad):
        a, = ctx.saved_tensors
        return np.full(a.data.shape, grad / a.data.size)

class Pow(autograd.Function):
    # a ^ n
    @staticmethod
    def forward(ctx, a, n):
        ctx.save_for_backward(a, n)
        return tensor.Tensor(a.data ** n.data)

    # y = a ^ n
    # dy/da = n * a
    @staticmethod
    def backward(ctx, grad):
        a, n = ctx.saved_tensors
        return grad * n.data * a.data, None

class LogSoftmax(autograd.Function):
    # log(e^a_i / sum(e^a)) = a_i - log(sum(e^a))
    @staticmethod
    def forward(ctx, a):
        ctx.save_for_backward(a)
        eps = np.finfo(float).eps
        x = a.data
        x = x - x.max(axis=1, keepdims=True)
        x = x - np.log(np.sum(np.exp(x), axis=1, keepdims=True) + eps)
        return tensor.Tensor(x)

    @staticmethod
    def backward(ctx, grad):
        a, = ctx.saved_tensors
        eps = np.finfo(float).eps
        x = a.data
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        sum_of_grad = np.sum(grad, axis=1, keepdims=True)
        return grad - x * sum_of_grad