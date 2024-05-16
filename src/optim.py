from time import time
import pennylane as qml
from pennylane import qchem
import pennylane.numpy as np
import matplotlib.pyplot as plt


class Optimizer:
    def __init__(self, eta, tol, maxiter, nqubits):
        self.eta = eta
        self.tol = tol
        self.maxiter = maxiter
        self.nqubits = nqubits

    def solve(self, weights):
        pass

    def optimize(self, func):

        q_delta = 0.001 * np.pi  # Initial spread of random quantum weights

        # initial weights for strongly entangling layers
        w = q_delta * np.random.randn(self.nqubits, requires_grad=True)

        # Gradient Descent Optimization Algorithm
        opt = qml.GradientDescentOptimizer(self.eta)

        # Optimization loop
        cost_history = []
        t0 = time()
        for it in range(self.maxiter):
            ta = time()
            w, cost_val = opt.step_and_cost(func, w)
            print("Step {:3d}    obj = {:9.7f}    time = {:9.7f} sec".format(it, cost_val, time() - ta))
            if np.abs(cost_val) < 1e-4:
                break
            cost_history.append(cost_val)
        print("\n Total Optimization Time: ", time() - t0, " sec")

        return w

    def plot_loss(self, loss_hist):
        plt.figure(1)
        plt.plot(loss_hist, "k", linewidth=2)
        plt.ylabel("Cost function")
        plt.xlabel("Optimization steps")


class GradientDescent_pennylane(Optimizer):
    def __init__(self, eta, tol, maxiter, nqubits):
        super().__init__(eta, tol, maxiter, nqubits)

    def vqc(self, weights):
        raise NotImplementedError("Not implemented yet.")


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


    solver = GradientDescent_pennylane(eta=0.8, tol=0.01, maxiter=10, nqubits=n)

    wopt = solver.optimize(cost, )

    print("wopt =", wopt)
