import pennylane as qml
from abc import ABC, abstractmethod


class Observable(ABC):

    def __init__(self):
        self.obs = None


class ObsPauliX(Observable):

    def __init__(self):
        super().__init__()
        self.obs = qml.PauliX


class ObsPauliY(Observable):

    def __init__(self):
        super().__init__()
        self.obs = qml.PauliY


class ObsPauliZ(Observable):

    def __init__(self):
        super().__init__()
        self.obs = qml.PauliZ
