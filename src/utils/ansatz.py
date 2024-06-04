import pennylane as qml
import pennylane.numpy as np
from abc import ABC, abstractmethod


class Ansatz(ABC):
    def __init__(self, nqubits, nlayers):
        self.nqubits = nqubits
        self.nlayers = nlayers
        self.nweights = None

    @abstractmethod
    def vqc(self, weights):
        pass

    @abstractmethod
    def init_weights(self):
        pass

    @abstractmethod
    def prep_weights(self, w):
        pass


class HardwareEfficient(Ansatz):
    def __init__(self, nqubits, nlayers):
        super().__init__(nqubits, nlayers)

    def vqc(self, weights):
        raise NotImplementedError("Not implemented yet.")

    def init_weights(self):
        raise NotImplementedError("Not implemented yet.")

    def prep_weights(self, w):
        pass


class StrongEntangling(Ansatz):

    def __init__(self, nqubits, nlayers):
        super().__init__(nqubits, nlayers)

    def vqc(self, weights):
        qml.StronglyEntanglingLayers(weights, wires=range(self.nqubits))

    def init_weights(self):
        self.nweights = self.nqubits * 3 * self.nlayers
        w = np.random.randn(self.nweights, requires_grad=True)
        return np.reshape(w, (self.nlayers, self.nqubits, 3))

    def prep_weights(self, w):
        return np.reshape(w, (self.nlayers, self.nqubits, 3))


class BasicEntangling(Ansatz):

    def __init__(self, nqubits, nlayers):
        super().__init__(nqubits, nlayers)

    def vqc(self, weights):
        qml.BasicEntanglerLayers(weights, wires=range(self.nqubits), rotation=qml.RY)

    def init_weights(self):
        self.nweights = self.nqubits * self.nlayers
        w = np.random.randn(self.nweights, requires_grad=True)
        return np.reshape(w, (self.nlayers, self.nqubits))

    def prep_weights(self, w):
        return np.reshape(w, (self.nlayers, self.nqubits))


class RotY(Ansatz):

    def __init__(self, nqubits, nlayers):
        super().__init__(nqubits, nlayers)

    def vqc(self, weights):

        for i in range(self.nqubits):
            qml.Hadamard(wires=i)

        for i in range(self.nqubits):
            qml.RY(weights[i], wires=i)

    def init_weights(self):
        self.nweights = self.nqubits
        w = np.random.randn(self.nweights, requires_grad=True)
        return np.reshape(w, (self.nqubits, 1))

    def prep_weights(self, w):
        return np.reshape(w, (self.nqubits, 1))


if __name__ == "__main__":

    hea = RotY(nqubits=2, nlayers=2)

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def qcirc(w):
        hea.vqc(w)
        return qml.state()


    print(qml.draw(qcirc)(np.ones(2)))
