from time import time
import pennylane as qml
import pennylane.numpy as np
from abc import ABC, abstractmethod


class OptimizerQML(ABC):
    def __init__(self, eta, tol, maxiter):
        self.eta = eta
        self.tol = tol
        self.maxiter = maxiter
        self.name = None

    def optimize(self, func, w):

        # Get the optimizer from the child class
        opt = self.get_optimizer()

        # Optimization loop
        cost_vals = []
        for it in range(self.maxiter):
            ta = time()
            w, cost_val = opt.step_and_cost(func, w)
            print("{:20s}     Step {:3d}    obj = {:9.7f}    time = {:9.7f} sec".format(self.name, it, cost_val, time() - ta))
            cost_vals.append(cost_val)
            if np.abs(cost_val) < self.tol:
                return w, cost_vals, it+1

        return w, cost_vals, self.maxiter

    @abstractmethod
    def get_optimizer(self):
        pass


class GradientDescentQML(OptimizerQML):
    def __init__(self, eta, tol, maxiter):
        super().__init__(eta, tol, maxiter)
        self.name = "GD"

    def get_optimizer(self):
        return qml.GradientDescentOptimizer(self.eta)


class AdamQML(OptimizerQML):
    def __init__(self, eta, tol, maxiter):
        super().__init__(eta, tol, maxiter)
        self.name = "Adam"

    def get_optimizer(self):
        return qml.AdamOptimizer(self.eta)


class AdagradQML(OptimizerQML):
    def __init__(self, eta, tol, maxiter):
        super().__init__(eta, tol, maxiter)
        self.name = "Adagrad"

    def get_optimizer(self):
        return qml.AdagradOptimizer(self.eta)


class MomentumQML(OptimizerQML):
    def __init__(self, eta, tol, maxiter, beta):
        super().__init__(eta, tol, maxiter)
        self.name = "Momentum"
        self.beta = beta

    def get_optimizer(self):
        return qml.MomentumOptimizer(self.eta)


class NesterovMomentumQML(OptimizerQML):
    def __init__(self, eta, tol, maxiter, beta):
        super().__init__(eta, tol, maxiter)
        self.name = "Nesterov"
        self.beta = beta

    def get_optimizer(self):
        return qml.NesterovMomentumOptimizer(self.eta)


class RMSPropQML(OptimizerQML):
    def __init__(self, eta, tol, maxiter):
        super().__init__(eta, tol, maxiter)
        self.name = "RMSProp"

    def get_optimizer(self):
        return qml.RMSPropOptimizer(self.eta)
