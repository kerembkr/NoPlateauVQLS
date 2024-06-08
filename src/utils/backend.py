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

    @abstractmethod
    def set_device(self, wires, diff_method=None, shots=None, seed='global', max_workers=None):
        """
        Abstract method to set the quantum device.
        """
        pass

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

    # @abstractmethod
    # def execute(self, circuit, meas_wires, *args, **kwargs):
    #     """
    #     Abstract method to execute a quantum circuit.
    #     """
    #     pass


class DefaultQubit(QDeviceBase):
    """
    Backend for the default.qubit simulator.
    """

    def __init__(self):
        super().__init__()
        self.name = "Default"

    def set_device(self, wires, diff_method=None, shots=None, seed='global', max_workers=None):
        try:
            self.qdevice = qml.device("default.qubit", wires=wires, shots=shots, seed=seed, max_workers=max_workers)
            print(
                f"Device set with {wires} wires, diff_method={diff_method}, shots={shots}, seed={seed}, max_workers={max_workers}")
        except Exception as e:
            print(f"Error setting device: {e}")
            self.qdevice = None

    # def execute(self, circuit, meas_wires, *args, **kwargs):
    #     if self.qdevice is None:
    #         raise ValueError("Device is not set. Call set_device() first.")
    #
    #     @qml.qnode(self.qdevice)
    #     def qnode(*qnode_args, **qnode_kwargs):
    #         circuit(*qnode_args, **qnode_kwargs)
    #         return self.returntype(self.observable(meas_wires))
    #
    #     try:
    #         result = qnode(*args, **kwargs)
    #         return result
    #     except Exception as e:
    #         print(f"Error executing circuit: {e}")
    #         return None


class LightningQubit(QDeviceBase):
    """
    Backend for the lightning.qubit simulator.
    """

    def __init__(self):
        super().__init__()
        self.name = "Lightning"

    def set_device(self, wires, diff_method=None, shots=None, seed='global', max_workers=None):
        try:
            self.qdevice = qml.device("lightning.qubit", wires=wires, shots=shots, seed=seed)
            print(f"Device set with {wires} wires, shots={shots}, seed={seed}")
        except Exception as e:
            print(f"Error setting device: {e}")
            self.qdevice = None

    # def execute(self, circuit, meas_wires, *args, **kwargs):
    #     if self.qdevice is None:
    #         raise ValueError("Device is not set. Call set_device() first.")
    #
    #     @qml.qnode(device=self.qdevice, interface=self.interface)
    #     def qnode(*qnode_args, **qnode_kwargs):
    #         circuit(*qnode_args, **qnode_kwargs)
    #         return self.returntype(self.observable(meas_wires))
    #
    #     try:
    #         result = qnode(*args, **kwargs)
    #         return result
    #     except Exception as e:
    #         print(f"Error executing circuit: {e}")
    #         return None


if __name__ == "__main__":

    def example_circuit(param1, param2, param3=None):
        qml.RX(param1, wires=0)
        qml.RY(param2, wires=0)
        if param3 is not None:
            qml.RZ(param3, wires=0)


    # Using DefaultQubit backend
    backend = DefaultQubit()
    backend.set_device(wires=1, diff_method="parameter-shift", shots=10)
    backend.set_observable(observable="sigz")
    backend.set_returntype(returntype="expval")
    if backend.qdevice is not None:
        print(f"Device successfully set: {backend.qdevice}")
    result = backend.execute(example_circuit, 0, 0.1, 0.2, param3=0.3)
    print(f"Result: {result}\n")

    # Using LightningQubit backend
    lightning_backend = LightningQubit()
    lightning_backend.set_device(wires=1, diff_method="backprop", shots=10)
    lightning_backend.set_observable(observable="sigz")
    lightning_backend.set_returntype(returntype="expval")
    if lightning_backend.qdevice is not None:
        print(f"Device successfully set: {lightning_backend.qdevice}")
    result = lightning_backend.execute(example_circuit, 0, 0.1, 0.2, param3=0.3)
    print(f"Result: {result}\n")
