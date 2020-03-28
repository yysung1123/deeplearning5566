import numpy as np
import dl56
import dl56.functional as F
import dl56.ops as ops

size = (2, 3)
t1 = dl56.Tensor(np.random.normal(size=size), requires_grad=True)
t2 = dl56.Tensor(np.random.normal(size=size))
print(t1.data)
print(t2.data)
t3 = t1 + t2
t4 = F.relu(t3)
t4.backward(np.ones(size))
print(t4.data)
print(t4.grad)
print(t3.grad)
print(t1.grad)