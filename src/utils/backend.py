import pennylane as qml
from abc import ABC, abstractmethod
from qiskit_aer import Aer

Aer.backends()


class QDeviceBase(ABC):
    """
    Abstract base class for quantum devices.
    """

    def __init__(self):
        self.name = None
        self.qdevice = None
        self.diff_method = None
        self.interface = None
        self.shots = None
        self.seed = None
        self.max_workers = None

    @abstractmethod
    def set_device(self, wires, diff_method=None, shots=None, seed='global', max_workers=None):
        """
        Abstract method to set the quantum device.
        """
        pass

    @abstractmethod
    def execute(self, circuit, *args, **kwargs):
        """
        Abstract method to execute a quantum circuit.
        """
        pass


class DefaultQubit(QDeviceBase):
    """
    Backend for the default.qubit simulator.
    """

    def __init__(self):
        super().__init__()
        self.name = "default.qubit"

    def set_device(self, wires, diff_method=None, shots=None, seed='global', max_workers=None):
        try:
            self.qdevice = qml.device("default.qubit", wires=wires, shots=shots, seed=seed, max_workers=max_workers)
            print(
                f"Device set with {wires} wires, diff_method={diff_method}, shots={shots}, seed={seed}, max_workers={max_workers}")
        except Exception as e:
            print(f"Error setting device: {e}")
            self.qdevice = None

    def execute(self, circuit, *args, **kwargs):
        if self.qdevice is None:
            raise ValueError("Device is not set. Call set_device() first.")

        @qml.qnode(self.qdevice)
        def qnode(*qnode_args, **qnode_kwargs):
            circuit(*qnode_args, **qnode_kwargs)
            return qml.expval(qml.PauliZ(0))

        try:
            result = qnode(*args, **kwargs)
            return result
        except Exception as e:
            print(f"Error executing circuit: {e}")
            return None


class LightningQubit(QDeviceBase):
    """
    Backend for the lightning.qubit simulator.
    """

    def __init__(self):
        super().__init__()
        self.name = "lightning.qubit"

    def set_device(self, wires, diff_method=None, shots=None, seed='global', max_workers=None):
        try:
            self.qdevice = qml.device("lightning.qubit", wires=wires, shots=shots, seed=seed)
            print(f"Device set with {wires} wires, shots={shots}, seed={seed}")
        except Exception as e:
            print(f"Error setting device: {e}")
            self.qdevice = None

    def execute(self, circuit, *args, **kwargs):
        if self.qdevice is None:
            raise ValueError("Device is not set. Call set_device() first.")

        @qml.qnode(self.qdevice)
        def qnode(*qnode_args, **qnode_kwargs):
            circuit(*qnode_args, **qnode_kwargs)
            return qml.expval(qml.PauliZ(0))

        try:
            result = qnode(*args, **kwargs)
            return result
        except Exception as e:
            print(f"Error executing circuit: {e}")
            return None


class DefaultQubitTorch(QDeviceBase):
    """
    Backend for the default.qubit simulator with PyTorch interface.
    """

    def __init__(self, analytic=None, torch_device=None):
        super().__init__()
        self.name = "default.qubit.torch"
        self.analytic = analytic
        self.torch_device = torch_device

    def set_device(self, wires, diff_method=None, shots=None, seed='global', max_workers=None):
        try:
            self.qdevice = qml.device("default.qubit.torch", wires=wires, shots=shots, analytic=self.analytic,
                                      torch_device=self.torch_device)
            print(
                f"Device set with {wires} wires, interface=torch, diff_method={diff_method}, shots={shots}, seed={seed}, max_workers={max_workers}")
        except Exception as e:
            print(f"Error setting device: {e}")
            self.qdevice = None

    def execute(self, circuit, *args, **kwargs):
        if self.qdevice is None:
            raise ValueError("Device is not set. Call set_device() first.")

        @qml.qnode(self.qdevice)
        def qnode(*qnode_args, **qnode_kwargs):
            circuit(*qnode_args, **qnode_kwargs)
            return qml.expval(qml.PauliZ(0))

        try:
            result = qnode(*args, **kwargs)
            return result
        except Exception as e:
            print(f"Error executing circuit: {e}")
            return None


class QiskitAer(QDeviceBase):
    """
    Backend for the Qiskit Aer simulator.
    """

    def __init__(self):
        super().__init__()
        self.name = "qiskit.aer"

    def set_device(self, wires, diff_method=None, shots=None, seed='global', max_workers=None):
        try:
            self.qdevice = qml.device("qiskit.aer", wires=wires, shots=shots, seed=seed, max_workers=max_workers)
            print(f"Device set with {wires} wires, shots={shots}, seed={seed}, max_workers={max_workers}")
        except Exception as e:
            print(f"Error setting device: {e}")
            self.qdevice = None

    def execute(self, circuit, *args, **kwargs):
        if self.qdevice is None:
            raise ValueError("Device is not set. Call set_device() first.")

        @qml.qnode(self.qdevice)
        def qnode(*qnode_args, **qnode_kwargs):
            circuit(*qnode_args, **qnode_kwargs)
            return qml.expval(qml.PauliZ(0))

        try:
            result = qnode(*args, **kwargs)
            return result
        except Exception as e:
            print(f"Error executing circuit: {e}")
            return None


if __name__ == "__main__":

    # Example usage
    def example_circuit(param1, param2, param3=None):
        qml.RX(param1, wires=0)
        qml.RY(param2, wires=0)
        if param3 is not None:
            qml.RZ(param3, wires=0)


    # Using DefaultQubit backend
    backend = DefaultQubit()
    backend.set_device(wires=1, diff_method="parameter-shift", shots=1000)

    # Verify that the device is set
    if backend.qdevice is not None:
        print(f"Device successfully set: {backend.qdevice}")
    #
    result = backend.execute(example_circuit, 0.1, 0.2, param3=0.3)
    print(f"Result: {result}")

    # Using LightningQubit backend
    lightning_backend = LightningQubit()
    lightning_backend.set_device(wires=1, diff_method="backprop", shots=1000)

    # Verify that the device is set
    if lightning_backend.qdevice is not None:
        print(f"Device successfully set: {lightning_backend.qdevice}")

    result = lightning_backend.execute(example_circuit, 0.1, 0.2, param3=0.3)
    print(f"Result: {result}")

    # Using DefaultQubitTorch backend
    torch_backend = DefaultQubitTorch()
    torch_backend.set_device(wires=1, diff_method="parameter-shift", shots=1000, seed=42, max_workers=2)

    # Verify that the device is set
    if torch_backend.qdevice is not None:
        print(f"Device successfully set: {torch_backend.qdevice}")

    result = torch_backend.execute(example_circuit, 0.1, 0.2, param3=0.3)
    print(f"Result: {result}")

    # Using QiskitAer backend
    aer_backend = QiskitAer()
    aer_backend.set_device(wires=1, shots=1000, seed=42, max_workers=2)

    # Verify that the device is set
    if aer_backend.qdevice is not None:
        print(f"Device successfully set: {aer_backend.qdevice}")

    result = aer_backend.execute(example_circuit, 0.1, 0.2, param3=0.3)
    print(f"Result: {result}")
