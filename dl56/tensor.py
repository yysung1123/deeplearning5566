from __future__ import annotations
from typing import Optional
from dl56.autograd import backward as autograd_backward
from dl56 import ops

import numpy as np

class Tensor:
    def __init__(self, data: np.ndarray, requires_grad: bool=False) -> None:
        self.data = data
        self.requires_grad = requires_grad
        self.grad = 0
        self.grad_prev = 0
        self.grad_fn = None
        self.children = []

    def __add__(self, other: Tensor) -> Tensor:
        # self + other
        return ops.Add.apply(self, other)

    def __sub__(self, other: Tensor) -> Tensor:
        # self - other
        return ops.Sub.apply(self, other)

    def __mul__(self, other: Tensor) -> Tensor:
        # self * other
        return ops.Mul.apply(self, other)

    def __neg__(self) -> Tensor:
        # -self
        return ops.Neg.apply(self)

    def __pow__(self, n) -> Tensor:
        # self ^ n
        if isinstance(n, (int, float)):
            n = Tensor(np.array([n]))
        if not isinstance(n, Tensor) or n.data.shape != (1,):
            raise NotImplementedError
        return ops.Pow.apply(self, n)

    def t(self) -> Tensor:
        # self.T
        return ops.Transpose.apply(self)

    def backward(self, grad: Optional[np.ndarray]=np.array([1])) -> None:
        if grad is not None:
            self.grad = grad
        autograd_backward(self)

    def set_grad_fn(self, fn) -> None:
        self.requires_grad = True
        self.grad_fn = fn

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    def add_grad(self, grad: np.ndarray) -> None:
        if self.data.shape != grad.shape:
            print(self.data.shape, grad.shape)
        if self.requires_grad:
            self.grad += grad

    def detach(self) -> Tensor:
        return Tensor(data=self.data)

    @property
    def leaf_node(self) -> bool:
        return len(self.children) == 0

    def matmul(self, other) -> Tensor:
        return ops.Matmul.apply(self, other)

    __module__ = 'dl56'