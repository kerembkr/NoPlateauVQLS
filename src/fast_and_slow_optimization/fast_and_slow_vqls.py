from time import time
import matplotlib.pyplot as plt
import pennylane as qml
import pennylane.numpy as np
import src.utils as utils
from skopt import gp_minimize
from scipy.optimize import differential_evolution


class VQLS:
    def __init__(self, A, b):
        self.nlayers = None
        self.A = A
        self.b = b
        self.nqubits = int(np.log(len(b)) / np.log(2))
        self.mats, self.wires, self.c = utils.get_paulis(self.A)

    def opt(self, epochs, eta=0.01):

        cost_history = []
        t0 = time()

        # fast optimization
        def print_progress(res_):
            iteration = len(res_.func_vals)
            print(f"Step {iteration}    obj = {res_.func_vals[-1]:9.7f}    params = {res_.x_iters[-1]}")

        epochs_bo = 10

        self.nlayers = 2

        nweights = self.nqubits * 3 * self.nlayers

        dimensions = [(-np.pi, +np.pi) for i in range(nweights)]

        res = gp_minimize(func=self.cost,
                          dimensions=dimensions,
                          callback=[print_progress],
                          acq_func="EI",
                          n_calls=epochs_bo)

        cost_history.append(res.func_vals.tolist())

        # initial guess for gradient optimizer
        w = np.tensor(res.x, requires_grad=True)

        # slow optimization
        opt = qml.GradientDescentOptimizer(eta)

        for it in range(epochs):
            ta = time()
            w, cost_val = opt.step_and_cost(self.cost, w)
            print("Step {:3d}    obj = {:9.7f}    time = {:9.7f} sec".format(it, cost_val, time() - ta))
            if np.abs(cost_val) < 1e-4:
                break
            cost_history.append(cost_val.item())
        print("\n Total Optimization Time: ", time() - t0, " sec")

        # make one single list
        cost_history = [item for sublist in cost_history for item in
                        (sublist if isinstance(sublist, list) else [sublist])]

        # Get the minimum value
        min_value = min(cost_history[0:epochs_bo])
        min_index = cost_history[0:epochs_bo].index(min_value)
        colors = ["y" if i == min_index else "g" if i < epochs_bo else "r" for i in range(len(cost_history))]

        plt.figure(1)
        plt.plot(cost_history, "k", linewidth=2)
        plt.scatter(range(len(cost_history)), cost_history, c=colors, linewidth=2)
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

        # for idx in range(self.nqubits):
        #     qml.Hadamard(wires=idx)
        #
        # for idx, element in enumerate(weights):
        #     qml.RY(element, wires=idx)

        weights = np.reshape(weights, (self.nlayers, self.nqubits, 3))

        qml.StronglyEntanglingLayers(weights=weights, wires=range(self.nqubits))

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
        try:
            return float(0.5 - 0.5 * abs(mu_sum) / (n_qubits * abs(psi_norm)))
        except:
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
    wopt = solver.opt(epochs=10, eta=1.0)
    xopt = solver.get_state(wopt)
