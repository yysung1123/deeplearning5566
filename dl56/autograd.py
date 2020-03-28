from typing import List

from . import util

def backward(tensors):
    tensors = reversed(topo_sort(tensors))
    for tensor in tensors:
        if not tensor.leaf_node:
            tensor.grad_fn(grad=tensor.grad)

def topo_sort(tensor) -> List:
    tensors_sorted = []
    tensors = set()
    def _topo_sort(tensor):
        for child in tensor.children:
            if child not in tensors:
                tensors.add(child)
                _topo_sort(child)
        tensors_sorted.append(tensor)

    _topo_sort(tensor)
    return tensors_sorted

class Function:
    @staticmethod
    def forward(ctx, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad):
        raise NotImplementedError

    @classmethod
    def backwardFn(cls, ctx, packed_args):
        def _Backward(grad):
            grads = cls.backward(ctx, grad)
            if not isinstance(grads, tuple):
                grads = (grads,)
            if len(grads) != len(packed_args):
                raise Exception
            for grad, tensor in zip(grads, packed_args):
                if grad is not None:
                    tensor.add_grad(grad)
        return _Backward

    @classmethod
    def apply(cls, *args, **kwargs):
        packed_args = util.pack_args(args, kwargs)
        tensor_args = [t for t in packed_args if util.is_tensor(t)]
        requires_grad = [t for t in tensor_args if t.requires_grad]
        ctx = Context()
        tensor = cls.forward(ctx, *args, **kwargs)
        if any(requires_grad):
            tensor.set_grad_fn(cls.backwardFn(ctx, packed_args))
            tensor.children.extend(requires_grad)
        return tensor

class Context:
    def save_for_backward(self, *tensors):
        self.to_save = tensors

    @property
    def saved_tensors(self):
        return self.to_save