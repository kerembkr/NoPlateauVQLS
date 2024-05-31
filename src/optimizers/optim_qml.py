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


if __name__ == "__main__":
    n = 6

    dev = qml.device("default.qubit", wires=n)


    @qml.qnode(dev)
    def cost(theta):
        hamiltonian, _ = qml.qchem.molecular_hamiltonian(["H", "H", "H"], np.array(
            [0.028, 0.054, 0.0, 0.986, 1.610, 0.0, 1.855, 0.002, 0.0]), charge=1)
        hf = qml.qchem.hf_state(electrons=2, orbitals=6)  # The Hartree-Fock State

        # Embedding
        qml.BasisState(hf, wires=range(n))

        # Parametrized Quantum Circuit
        qml.DoubleExcitation(theta[0], wires=[0, 1, 2, 3])
        qml.DoubleExcitation(theta[1], wires=[0, 1, 4, 5])

        return qml.expval(hamiltonian)  # <H>


    eta = 0.8
    tol = 0.01
    maxiter = 50

    solver1 = GradientDescentQML(eta=eta, tol=tol, maxiter=maxiter, nqubits=n)
    solver2 = AdamQML(eta=eta, tol=tol, maxiter=maxiter, nqubits=n)
    solver3 = AdagradQML(eta=eta, tol=tol, maxiter=maxiter, nqubits=n)
    solver4 = MomentumQML(eta=eta, tol=tol, maxiter=maxiter, nqubits=n)
    solver5 = NesterovMomentumQML(eta=eta, tol=tol, maxiter=maxiter, nqubits=n)
    solver6 = RMSPropQML(eta=eta, tol=tol, maxiter=maxiter, nqubits=n)

    _, hist1 = solver1.optimize(cost)
    _, hist2 = solver2.optimize(cost)
    _, hist3 = solver3.optimize(cost)
    _, hist4 = solver4.optimize(cost)
    _, hist5 = solver5.optimize(cost)
    _, hist6 = solver6.optimize(cost)

    plot_loss(hist1, "GD")
    plot_loss(hist2, "Adam")
    plot_loss(hist3, "Adagrad")
    plot_loss(hist4, "Momentum")
    plot_loss(hist5, "Nesterov")
    plot_loss(hist6, "RMSProp")
    plt.show()


