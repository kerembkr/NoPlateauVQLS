import pennylane as qml
import src.utils.utils as utils
import pennylane.numpy as np
from skopt import gp_minimize
from src.utils.ansatz import StrongEntangling, BasicEntangling, HardwareEfficient, RotY
from src.optimizers.optim_qml import AdamQML, AdagradQML, GradientDescentQML, MomentumQML, NesterovMomentumQML, \
    RMSPropQML
import matplotlib.pyplot as plt


class VQLS:
    def __init__(self, A, b):

        # linear system
        self.A = A
        self.b = b

        # number of qubits
        self.nqubits = int(np.log(len(b)) / np.log(2))

        # Pauli decomposition
        self.mats, self.wires, self.c = utils.get_paulis(self.A)

    def opt(self, optimizer=None, ansatz=None, epochs=100, epochs_bo=None, tol=1e-4):

        if optimizer is None:
            optimizer = GradientDescentQML()

        if ansatz is None:
            self.ansatz = StrongEntangling(nqubits=self.nqubits, nlayers=1)
        else:
            self.ansatz = ansatz

        # initial weights
        w = self.ansatz.init_weights()

        # global optimization
        if epochs_bo is not None:
            w, _ = self.bayesopt_init(epochs_bo=epochs_bo)

        # local optimization
        w, cost_vals, iters = optimizer.optimize(func=self.cost, w=w, epochs=epochs, tol=tol)

        return w, cost_vals

    def qlayer(self, l=None, lp=None, j=None, part=None):

        dev_mu = qml.device("default.qubit", wires=nqubits + 1)

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
            qml.Hadamard(wires=nqubits)

            # For estimating the imaginary part of the coefficient "mu", we must add a "-i"
            # phase gate.
            if part == "Im" or part == "im":
                qml.PhaseShift(-np.pi / 2, wires=nqubits)

            # Variational circuit generating a guess for the solution vector |x>
            self.V(weights)

            # Controlled application of the unitary component A_l of the problem matrix A.
            self.CA(l, self.mats, self.wires)

            # Adjoint of the unitary U_b associated to the problem vector |b>.
            qml.adjoint(self.U_b)(self.b)

            # Controlled Z operator at position j. If j = -1, apply the identity.
            if j != -1:
                qml.CZ(wires=[nqubits, j])

            # Unitary U_b associated to the problem vector |b>.
            self.U_b(self.b)

            # Controlled application of Adjoint(A_lp).
            qml.adjoint(self.CA)(lp, self.mats, self.wires)

            # Second Hadamard gate applied to the ancillary qubit.
            qml.Hadamard(wires=nqubits)

            # Expectation value of Z for the ancillary qubit.
            return qml.expval(qml.PauliZ(wires=nqubits))

        return qcircuit

    def bayesopt_init(self, epochs_bo=10):

        def print_progress(res_):
            print("{:20s}    Step {:3d}    obj = {:9.7f} ".format(
                    "Bayesian Optimization", len(res_.func_vals), res_.func_vals[-1]))

        # set parameter space
        dimensions = [(-np.pi, +np.pi) for _ in range(self.ansatz.nweights)]

        print(np.shape(dimensions))

        print(dimensions)

        # bayesian optimization
        res = gp_minimize(func=self.cost,
                          dimensions=dimensions,
                          callback=[print_progress],
                          acq_func="EI",
                          n_calls=epochs_bo)

        # save cost function values
        cost_hist_bo = res.func_vals.tolist()

        # initial guess for gradient optimizer
        w = np.tensor(res.x, requires_grad=True)

        # reshape weights
        w = self.ansatz.prep_weights(w)

        return w, cost_hist_bo

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

        weights = self.ansatz.prep_weights(weights)

        # apply unitary ansatz
        self.ansatz.vqc(weights=weights)

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
                for j in range(0, nqubits):
                    mu_real_qnode = self.qlayer(l=l, lp=lp, j=j, part="Re")
                    mu_imag_qnode = self.qlayer(l=l, lp=lp, j=j, part="Im")
                    mu_real = mu_real_qnode(weights)
                    mu_imag = mu_imag_qnode(weights)

                    mu_sum += self.c[l] * np.conj(self.c[lp]) * (mu_real + 1.0j * mu_imag)

        # Cost function C_L
        try:
            return float(0.5 - 0.5 * abs(mu_sum) / (nqubits * abs(psi_norm)))
        except:
            return 0.5 - 0.5 * abs(mu_sum) / (nqubits * abs(psi_norm))

    def solve_classic(self):
        return np.linalg.solve(self.A, self.b)

    def get_state(self, params):

        # classical probabilities
        A_inv = np.linalg.inv(self.A)
        x = np.dot(A_inv, self.b)
        c_probs = (x / np.linalg.norm(x)) ** 2

        # quantum probabilities
        n_shots = 10 ** 6
        dev_x = qml.device("lightning.qubit", wires=nqubits, shots=n_shots)

        @qml.qnode(dev_x, interface="autograd")
        def prepare_and_sample(weights):
            self.V(weights)
            return qml.sample()

        raw_samples = prepare_and_sample(params)
        if nqubits == 1:
            raw_samples = [[_] for _ in raw_samples]
        samples = []
        for sam in raw_samples:
            samples.append(int("".join(str(bs) for bs in sam), base=2))
        q_probs = np.round(np.bincount(samples) / n_shots, 2)

        # plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
        ax1.bar(np.arange(0, 2 ** nqubits), c_probs, color="skyblue")
        ax1.set_xlim(-0.5, 2 ** nqubits - 0.5)
        ax1.set_ylim(0.0, 1.0)
        ax1.set_xlabel("Vector space basis")
        ax1.set_title("Classical probabilities")
        ax2.bar(np.arange(0, 2 ** nqubits), q_probs, color="plum")
        ax2.set_xlim(-0.5, 2 ** nqubits - 0.5)
        ax2.set_ylim(0.0, 1.0)
        ax2.set_xlabel("Hilbert space basis")
        ax2.set_title("Quantum probabilities")
        plt.show()

        dev_x = qml.device("lightning.qubit", wires=nqubits, shots=None)

        @qml.qnode(dev_x)
        def prepare_and_get_state(weights):
            self.V(weights)  # V(weight)|0>
            return qml.state()  # |x>

        state = np.round(np.real(prepare_and_get_state(params)), 2)

        print(" x  =", np.round(x / np.linalg.norm(x), 2))
        print("|x> =", state)

        return state


if __name__ == "__main__":

    # reproducibility
    np.random.seed(42)

    # number of qubits & layers
    nqubits = 1
    nlayers = 2

    epochs = 10

    # random symmetric positive definite matrix
    A0, b0 = utils.get_random_ls(nqubits, easy_example=True)

    # init
    solver = VQLS(A=A0, b=b0)

    # choose optimizer
    optims = [GradientDescentQML(),
              AdamQML(),
              AdagradQML(),
              MomentumQML(),
              NesterovMomentumQML(),
              RMSPropQML()]

    ansatz_ = StrongEntangling(nqubits=nqubits, nlayers=nlayers)

    cost_hists = {}

    for optim in optims:
        wopt, cost_hist = solver.opt(optimizer=optim,
                                     ansatz=ansatz_,
                                     epochs=epochs,
                                     epochs_bo=10,
                                     tol=1e-6)

        cost_hists[optim.name] = cost_hist

    title = "{:s}    qubits = {:d}    layers = {:d}".format(ansatz_.__class__.__name__, nqubits, nlayers)
    utils.plot_costs(data=cost_hists, save_png=True, title=title)

    plt.show()
