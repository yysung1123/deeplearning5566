class Optim:
    def __init__(self, params):
        self.params = params

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

class SGD(Optim):
    def __init__(self, params, lr, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        super().__init__(params)

    def step(self):
        for param in self.params:
            v = param.grad * self.lr + param.grad_prev * self.momentum
            param.data -= v
            param.grad_prev = v