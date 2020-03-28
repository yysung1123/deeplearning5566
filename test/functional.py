import numpy as np
import dl56
import dl56.nn as nn
import dl56.functional as F

X = dl56.Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
Y = dl56.Tensor(np.array([[0], [1], [1], [0]]))
w1 = nn.Parameter(np.array([[1, 1], [1, 1]]))
b1 = nn.Parameter(np.array([0, -1]))
l1 = F.linear(X, w1, b1)
print(l1.data)
a1 = F.relu(l1)
w2 = nn.Parameter(np.array([[1, -2]]))
print(a1.data.shape)
l2 = F.linear(a1, w2)
print(l2.data)
loss = F.mse_loss(l2, Y)
print(loss.data)
loss.backward()
print(b1.grad)

t = dl56.Tensor(np.random.normal(size=(4, 3)), requires_grad=True)
out = F.log_softmax(t)
out.backward(np.ones((4,3)))
print(np.exp(out.data).sum(axis=1))

t = dl56.Tensor(np.random.normal(size=(4,3)), requires_grad=True)