import pennylane as qml
from abc import ABC, abstractmethod


class QDeviceBase(ABC):
    """
    Abstract base class for quantum devices.
    """

    def __init__(self):
        self.name = None
        self.qdevice = None

    @abstractmethod
    def set_device(self, nwires):
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
        self.qdevice = None

    def set_device(self, nwires):
        try:
            self.qdevice = qml.device("default.qubit", wires=nwires)
            print(f"Device set with {nwires} wires")
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


class DefaultMixed(QDeviceBase):
    """
    Backend for the default.mixed simulator.
    """

    def __init__(self):
        super().__init__()
        self.name = "default.mixed"
        self.qdevice = None

    def set_device(self, nwires):
        try:
            self.qdevice = qml.device("default.mixed", wires=nwires)
            print(f"Device set with {nwires} wires")
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


class DefaultGaussian(QDeviceBase):
    """
    Backend for the default.gaussian simulator.
    """

    def __init__(self):
        super().__init__()
        self.name = "default.gaussian"
        self.qdevice = None

    def set_device(self, nwires):
        try:
            self.qdevice = qml.device("default.gaussian", wires=nwires)
            print(f"Device set with {nwires} wires")
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
        self.qdevice = None

    def set_device(self, nwires):
        try:
            self.qdevice = qml.device("lightning.qubit", wires=nwires)
            print(f"Device set with {nwires} wires")
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


class LightningGPU(QDeviceBase):
    """
    Backend for the lightning.gpu simulator.
    """

    def __init__(self):
        super().__init__()
        self.name = "lightning.gpu"
        self.qdevice = None

    def set_device(self, nwires):
        try:
            self.qdevice = qml.device("lightning.gpu", wires=nwires)
            print(f"Device set with {nwires} wires")
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


class LightningKokkos(QDeviceBase):
    """
    Backend for the lightning.kokkos simulator.
    """

    def __init__(self):
        super().__init__()
        self.name = "lightning.kokkos"
        self.qdevice = None

    def set_device(self, nwires):
        try:
            self.qdevice = qml.device("lightning.kokkos", wires=nwires)
            print(f"Device set with {nwires} wires")
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
    backend.set_device(nwires=1)

    # Verify that the device is set
    if backend.qdevice is not None:
        print(f"Device successfully set: {backend.qdevice}")

    result = backend.execute(example_circuit, 0.1, 0.2, param3=0.3)
    print(f"Result: {result}")

    # Using LightningGPU backend
    gpu_backend = LightningQubit()
    gpu_backend.set_device(nwires=1)

    # Verify that the device is set
    if gpu_backend.qdevice is not None:
        print(f"Device successfully set: {gpu_backend.qdevice}")

    result = gpu_backend.execute(example_circuit, 0.1, 0.2, param3=0.3)
    print(f"Result: {result}")
