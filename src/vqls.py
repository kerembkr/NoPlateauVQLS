from time import time
import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
import utils


class VQLS:
    def __init__(self, A, b):
        self.A = A
        self.b = b
        self.nqubits = int(np.log(len(b))/np.log(2))

        self.mats, self.wires, self.c = utils.get_paulis(self.A)

    def opt(self):
        # optimization configs
        steps = 5  # Number of optimization steps
        eta = 0.8  # Learning rate
        q_delta = 0.001 * np.pi  # Initial spread of random quantum weights
        np.random.seed(0)
        # layers = 1

        # initial weights for strongly entangling layers
        w = q_delta * np.random.randn(n_qubits, requires_grad=True)

        # Gradient Descent Optimization Algorithm
        opt = qml.GradientDescentOptimizer(eta)

        # Optimization loop
        cost_history = []
        t0 = time()
        for it in range(steps):
            ta = time()
            w, cost_val = opt.step_and_cost(self.cost, w)
            print("Step {:3d}    obj = {:9.7f}    time = {:9.7f} sec".format(it, cost_val, time() - ta))
            if np.abs(cost_val) < 1e-4:
                break
            cost_history.append(cost_val)
        print("\n Total Optimization Time: ", time() - t0, " sec")

        plt.figure(1)
        plt.plot(cost_history, "k", linewidth=2)
        plt.ylabel("Cost function")
        plt.xlabel("Optimization steps")

        return w

    def qlayer(self, l=None, lp=None, j=None, part=None):

        dev_mu = qml.device("default.qubit", wires=n_qubits + 1)

        @qml.qnode(dev_mu)
        def qcircuit(weights):
            """
            Variational circuit mapping the ground state |0> to the ansatz state |x>.

            Args:
                vec (np.array): Vector state to be embedded in a quantum state.

            Returns:
                Expectation value of ancilla qubit in Pauli-Z basis
                :param weights:
            """

            # First Hadamard gate applied to the ancillary qubit.
            qml.Hadamard(wires=n_qubits)

            # For estimating the imaginary part of the coefficient "mu", we must add a "-i"
            # phase gate.
            if part == "Im" or part == "im":
                qml.PhaseShift(-np.pi / 2, wires=n_qubits)

            # Variational circuit generating a guess for the solution vector |x>
            self.V(weights)

            # Controlled application of the unitary component A_l of the problem matrix A.
            self.CA(l, self.mats, self.wires)

            # Adjoint of the unitary U_b associated to the problem vector |b>.
            qml.adjoint(self.U_b)(self.b)

            # Controlled Z operator at position j. If j = -1, apply the identity.
            if j != -1:
                qml.CZ(wires=[n_qubits, j])

            # Unitary U_b associated to the problem vector |b>.
            self.U_b(self.b)

            # Controlled application of Adjoint(A_lp).
            qml.adjoint(self.CA)(lp, self.mats, self.wires)

            # Second Hadamard gate applied to the ancillary qubit.
            qml.Hadamard(wires=n_qubits)

            # Expectation value of Z for the ancillary qubit.
            return qml.expval(qml.PauliZ(wires=n_qubits))

        return qcircuit

    def U_b(self, vec):
        """
        Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>.
        """
        qml.AmplitudeEmbedding(features=vec, wires=range(self.nqubits), normalize=True)  # O(n^2)

    def CA(self, idx, matrices, qubits):
        """
        Controlled versions of the unitary components A_l of the problem matrix A.
        """
        qml.ControlledQubitUnitary(matrices[idx], control_wires=[self.nqubits], wires=qubits[idx])

    def V(self, weights):
        """
        Variational circuit mapping the ground state |0> to the ansatz state |x>.

        """

        for idx in range(self.nqubits):
            qml.Hadamard(wires=idx)

        for idx, element in enumerate(weights):
            qml.RY(element, wires=idx)

    def cost(self, weights):
        """
      Local version of the cost function. Tends to zero when A|x> is proportional to |b>.

      Args:
          weights (np.array): trainable parameters for the variational circuit.

      Returns:
          Cost function value (float)

      """

        mu_sum = 0.0
        psi_norm = 0.0
        for l in range(0, len(self.c)):
            for lp in range(0, len(self.c)):
                psi_real_qnode = self.qlayer(l=l, lp=lp, j=-1, part="Re")
                psi_imag_qnode = self.qlayer(l=l, lp=lp, j=-1, part="Im")
                psi_real = psi_real_qnode(weights)
                psi_imag = psi_imag_qnode(weights)

                psi_norm += self.c[l] * np.conj(self.c[lp]) * (psi_real + 1.0j * psi_imag)
                for j in range(0, n_qubits):
                    mu_real_qnode = self.qlayer(l=l, lp=lp, j=j, part="Re")
                    mu_imag_qnode = self.qlayer(l=l, lp=lp, j=j, part="Im")
                    mu_real = mu_real_qnode(weights)
                    mu_imag = mu_imag_qnode(weights)

                    mu_sum += self.c[l] * np.conj(self.c[lp]) * (mu_real + 1.0j * mu_imag)

        # Cost function C_L
        return 0.5 - 0.5 * abs(mu_sum) / (n_qubits * abs(psi_norm))

    def solve_classic(self):
        return np.linalg.solve(self.A, self.b)

    def get_state(self, params):

        # classical probabilities
        A_inv = np.linalg.inv(self.A)
        x = np.dot(A_inv, self.b)
        c_probs = (x / np.linalg.norm(x)) ** 2

        # quantum probabilities
        n_shots = 10 ** 6
        dev_x = qml.device("lightning.qubit", wires=n_qubits, shots=n_shots)

        @qml.qnode(dev_x, interface="autograd")
        def prepare_and_sample(weights):
            self.V(weights)
            return qml.sample()

        raw_samples = prepare_and_sample(params)
        if n_qubits == 1:
            raw_samples = [[_] for _ in raw_samples]
        samples = []
        for sam in raw_samples:
            samples.append(int("".join(str(bs) for bs in sam), base=2))
        q_probs = np.round(np.bincount(samples) / n_shots, 2)

        # plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4))
        ax1.bar(np.arange(0, 2 ** n_qubits), c_probs, color="skyblue")
        ax1.set_xlim(-0.5, 2 ** n_qubits - 0.5)
        ax1.set_ylim(0.0, 1.0)
        ax1.set_xlabel("Vector space basis")
        ax1.set_title("Classical probabilities")
        ax2.bar(np.arange(0, 2 ** n_qubits), q_probs, color="plum")
        ax2.set_xlim(-0.5, 2 ** n_qubits - 0.5)
        ax2.set_ylim(0.0, 1.0)
        ax2.set_xlabel("Hilbert space basis")
        ax2.set_title("Quantum probabilities")
        plt.show()

        dev_x = qml.device("lightning.qubit", wires=n_qubits, shots=None)

        @qml.qnode(dev_x)
        def prepare_and_get_state(weights):
            self.V(weights)  # V(weight)|0>
            return qml.state()  # |x>

        state = np.round(np.real(prepare_and_get_state(params)), 2)

        print(" x  =", np.round(x / np.linalg.norm(x), 2))
        print("|x> =", state)

        return state


if __name__ == "__main__":

    # number of qubits
    n_qubits = 2

    # matrix
    A0 = np.eye(2 ** n_qubits, 2 ** n_qubits)
    A0[0, 0] = 2.0

    # vector
    b0 = np.ones(2 ** n_qubits)
    b0 = b0 / np.linalg.norm(b0)

    # init
    solver = VQLS(A=A0, b=b0)

    # get solution of lse
    wopt = solver.opt()
    xopt = solver.get_state(wopt)
