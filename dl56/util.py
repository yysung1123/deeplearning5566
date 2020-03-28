import dl56

def maybe_unexpand(grad, old_size):
    num_unsqueezed = grad.ndim - len(old_size)
    for _ in range(num_unsqueezed):
        grad = grad.sum(0, keepdims=False)
    return grad

def pack_args(args, kwargs):
    return tuple(list(args) + list(kwargs.values()))

def is_tensor(t):
    return isinstance(t, dl56.Tensor)