import pennylane as qml
import pennylane.numpy as np


class Ansatz:
    def __init__(self, nqubits, nlayers):
        self.nqubits = nqubits
        self.nlayers = nlayers

    def vqc(self, weights):
        pass

    def prepare_weights(self, weights):
        pass


class HardwareEfficient(Ansatz):
    def __init__(self, nqubits, nlayers):
        super().__init__(nqubits, nlayers)

    def vqc(self, weights):
        raise NotImplementedError("Not implemented yet.")


class StrongEntangling(Ansatz):

    def __init__(self, nqubits, nlayers):
        super().__init__(nqubits, nlayers)

    def vqc(self, weights):
        qml.StronglyEntanglingLayers(weights, wires=range(self.nqubits))

    def prepare_weights(self, weights):
        return np.reshape(weights, (self.nlayers, self.nqubits, 3))


class BasicEntangling(Ansatz):

    def __init__(self, nqubits, nlayers):
        super().__init__(nqubits, nlayers)

    def vqc(self, weights):
        qml.BasicEntanglerLayers(weights, wires=range(self.nqubits))


class RotY(Ansatz):

    def __init__(self, nqubits, nlayers):
        super().__init__(nqubits, nlayers)

    def vqc(self, weights):

        for i in range(self.nqubits):
            qml.Hadamard(wires=i)

        for i in range(self.nqubits):
            qml.RY(weights[i], wires=i)


if __name__ == "__main__":
    hea = RotY(nqubits=2, nlayers=2)

    dev = qml.device("default.qubit", wires=2)

    @qml.qnode(dev)
    def qcirc(w):
        hea.vqc(w)
        return qml.state()

    print(qml.draw(qcirc)(np.ones(2)))
