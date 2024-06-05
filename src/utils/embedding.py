import pennylane as qml
import pennylane.numpy as np
from abc import ABC, abstractmethod


class StatePrep(ABC):
    def __init__(self, wires):
        self.wires = wires

    @abstractmethod
    def prep(self, weights):
        pass


class AmplitudeEmbedding(StatePrep):
    def __init__(self, wires):
        super().__init__(wires)

    def prep(self, vec):
        qml.AmplitudeEmbedding(vec, self.wires, normalize=True)


class MottonenStatePrep(StatePrep):

    def __init__(self, wires):
        super().__init__(wires)

    def prep(self, vec):
        qml.MottonenStatePreparation(vec/np.linalg.norm(vec), self.wires)


if __name__ == "__main__":

    nqubits = 2
    qubits = range(2)
    b = np.ones(2**nqubits)
    # emb = AmplitudeEmbedding(wires=qubits)
    emb = MottonenStatePrep(wires=qubits)

    dev = qml.device("default.qubit", wires=nqubits)

    @qml.qnode(dev)
    def qcirc(vec):
        emb.prep(vec)
        return qml.state()

    print(qcirc(b))
