from time import time
import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print()
        print(f"Function '{func.__name__}' total computation time: {end_time - start_time:.7f} seconds")
        print()
        return result

    return wrapper


def plot_loss(loss_hist, label):
    # plt.figure(1)
    # plt.plot(loss_hist, "k", linewidth=2)
    plt.plot(loss_hist, linewidth=2, label=label)

    plt.ylabel("Cost function")
    plt.xlabel("Optimization steps")
    plt.legend()
    # plt.show()


class OptimizerQML(ABC):
    def __init__(self, eta, tol, maxiter, nqubits):
        self.eta = eta
        self.tol = tol
        self.maxiter = maxiter
        self.nqubits = nqubits

    @timing_decorator
    def optimize(self, func):
        # Initial weights for strongly entangling layers
        w = 1.0 * np.random.randn(self.nqubits, requires_grad=True)

        # Get the optimizer from the child class
        opt = self.get_optimizer()

        # Optimization loop
        cost_vals = []
        for it in range(self.maxiter):
            ta = time()
            w, cost_val = opt.step_and_cost(func, w)
            print("Step {:3d}    obj = {:9.7f}    time = {:9.7f} sec".format(it, cost_val, time() - ta))
            if np.abs(cost_val) < self.tol:
                break
            cost_vals.append(cost_val)

        return w, cost_vals

    @abstractmethod
    def get_optimizer(self):
        pass


class GradientDescentQML(OptimizerQML):
    def __init__(self, eta, tol, maxiter, nqubits):
        super().__init__(eta, tol, maxiter, nqubits)

    def get_optimizer(self):
        return qml.GradientDescentOptimizer(self.eta)


class AdamQML(OptimizerQML):
    def __init__(self, eta, tol, maxiter, nqubits):
        super().__init__(eta, tol, maxiter, nqubits)

    def get_optimizer(self):
        return qml.AdamOptimizer(self.eta)


class AdagradQML(OptimizerQML):
    def __init__(self, eta, tol, maxiter, nqubits):
        super().__init__(eta, tol, maxiter, nqubits)

    def get_optimizer(self):
        return qml.AdagradOptimizer(self.eta)


class MomentumQML(OptimizerQML):
    def __init__(self, eta, tol, maxiter, nqubits):
        super().__init__(eta, tol, maxiter, nqubits)

    def get_optimizer(self):
        return qml.MomentumOptimizer(self.eta)


class NesterovMomentumQML(OptimizerQML):
    def __init__(self, eta, tol, maxiter, nqubits):
        super().__init__(eta, tol, maxiter, nqubits)

    def get_optimizer(self):
        return qml.NesterovMomentumOptimizer(self.eta)


class RMSPropQML(OptimizerQML):
    def __init__(self, eta, tol, maxiter, nqubits):
        super().__init__(eta, tol, maxiter, nqubits)

    def get_optimizer(self):
        return qml.RMSPropOptimizer(self.eta)
