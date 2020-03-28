import dl56
import dl56.functional as F
import dl56.nn as nn
import dl56.optim as optim
import numpy as np

X = dl56.Tensor(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]))
Y = dl56.Tensor(np.array([[0], [1], [1], [0]]))

class Net(nn.Module):
    def __init__(self):
        self.fc1 = nn.Linear(2, 2, bias=True)
        self.fc2 = nn.Linear(2, 1, bias=False)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)

epochs = 1000
learning_rate = 1e-1
np.random.seed(0)
model = Net()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.0)

for epoch in range(epochs):
    out = model.forward(X)
    loss = F.mse_loss(out, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(epoch, loss.data)
        print(model.forward(X).data)