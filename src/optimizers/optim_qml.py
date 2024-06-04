from time import time
import pennylane as qml
import pennylane.numpy as np
from abc import ABC, abstractmethod


class OptimizerQML(ABC):
    def __init__(self):
        self.name = None

    def optimize(self, func, w, epochs, tol):

        # Get the optimizer from the child class
        opt = self.get_optimizer()

        # Optimization loop
        cost_vals = []
        for it in range(epochs):
            ta = time()
            w, cost_val = opt.step_and_cost(func, w)
            print("{:20s}     Step {:3d}    obj = {:9.7f}    time = {:9.7f} sec".format(self.name, it, cost_val, time() - ta))
            cost_vals.append(cost_val)
            if np.abs(cost_val) < tol:
                return w, cost_vals, it+1

        return w, cost_vals, epochs

    @abstractmethod
    def get_optimizer(self):
        pass


class GradientDescentQML(OptimizerQML):
    def __init__(self, eta=0.1):
        super().__init__()
        self.eta = eta
        self.name = "GD"

    def get_optimizer(self):
        return qml.GradientDescentOptimizer(self.eta)


class AdamQML(OptimizerQML):
    def __init__(self, eta=0.1, beta1=0.9, beta2=0.99, eps=1e-8):
        super().__init__()
        self.eta = eta
        self.name = "Adam"
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def get_optimizer(self):
        return qml.AdamOptimizer(stepsize=self.eta, beta1=self.beta1, beta2=self.beta2, eps=self.eps)


class AdagradQML(OptimizerQML):
    def __init__(self, eta=0.1, eps=1e-8):
        super().__init__()
        self.eta = eta
        self.name = "Adagrad"
        self.eps = eps

    def get_optimizer(self):
        return qml.AdagradOptimizer(stepsize=self.eta, eps=self.eps)


class MomentumQML(OptimizerQML):
    def __init__(self, eta=0.1, beta=0.9):
        super().__init__()
        self.eta = eta
        self.name = "Momentum"
        self.beta = beta

    def get_optimizer(self):
        return qml.MomentumOptimizer(stepsize=self.eta, momentum=self.beta)


class NesterovMomentumQML(OptimizerQML):
    def __init__(self, eta=0.1, beta=0.9):
        super().__init__()
        self.eta = eta
        self.name = "Nesterov"
        self.beta = beta

    def get_optimizer(self):
        return qml.NesterovMomentumOptimizer(stepsize=self.eta, momentum=self.beta)


class RMSPropQML(OptimizerQML):
    def __init__(self, eta=0.1, decay=0.9, eps=1e-8):
        super().__init__()
        self.eta = eta
        self.name = "RMSProp"
        self.decay = decay
        self.eps = eps

    def get_optimizer(self):
        return qml.RMSPropOptimizer(stepsize=self.eta, decay=self.decay, eps=self.eps)
