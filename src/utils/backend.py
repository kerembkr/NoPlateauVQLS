import pennylane as qml
from abc import ABC, abstractmethod


class QDeviceBase(ABC):
    """
    Abstract base class for quantum devices.
    """

    def __init__(self):

        # device
        self.name = None
        self.qdevice = None
        self.diff_method = None
        self.interface = None
        self.shots = None
        self.seed = None
        self.max_workers = None

        # measurement
        self.observable = None
        self.returntype = None

        self.observable_list = {
            "sigx": qml.PauliX,
            "sigy": qml.PauliY,
            "sigz": qml.PauliZ
        }

        self.returntype_list = {
            "expval": qml.expval,
            "probs": qml.probs,
            "state": qml.state,
            "counts": qml.counts,
            "sample": qml.sample
        }

    def set_device(self, device_name, **kwargs):
        try:
            self.qdevice = qml.device(device_name, **kwargs)
            kwargs_str = ', '.join(f"{key}={value}" for key, value in kwargs.items())
            print(f"Device set with {kwargs_str}")
        except Exception as e:
            print(f"Error setting device: {e}")
            self.qdevice = None

    def set_observable(self, observable):
        """
        Method to set the observable.
        """

        self.observable = self.observable_list[observable]

    def set_returntype(self, returntype):
        """
        Method to set the return Type.
        """

        self.returntype = self.returntype_list[returntype]


class DefaultQubit(QDeviceBase):
    """
    Backend for the default.qubit simulator.
    """

    def __init__(self, wires, shots=None, seed=None, max_workers=None, observable="sigz", returntype="expval"):
        super().__init__()
        self.name = "default.qubit"
        self.wires = wires
        self.shots = shots
        self.seed = seed
        self.max_workers = max_workers
        self.observable = self.observable_list[observable]
        self.returntype = self.returntype_list[returntype]

        # set the backend device
        self.set_device(device_name=self.name,
                        wires=self.wires,
                        shots=self.shots,
                        seed=self.seed,
                        max_workers=self.max_workers)


class LightningQubit(QDeviceBase):
    """
    Backend for the lightning.qubit simulator.
    """

    def __init__(self, wires, shots=None, seed=None, interface="auto", observable="sigz", returntype="expval"):
        super().__init__()
        self.name = "lightning.qubit"
        self.wires = wires
        self.shots = shots
        self.seed = seed
        self.interface = interface
        self.observable = self.observable_list[observable]
        self.returntype = self.returntype_list[returntype]

        # set the backend device
        self.set_device(device_name=self.name,
                        wires=self.wires,
                        shots=self.shots,
                        seed=self.seed)


if __name__ == "__main__":

    # Using DefaultQubit backend
    backend = DefaultQubit(wires=[0, 1])
    if backend.qdevice is not None:
        print(f"Device successfully set: {backend.qdevice}")

    # Using LightningQubit backend
    lightning_backend = LightningQubit(wires=[0, 1])
    if lightning_backend.qdevice is not None:
        print(f"Device successfully set: {lightning_backend.qdevice}")
