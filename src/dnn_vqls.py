import torch
import functools
import torch.nn as nn
from time import time
import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt

n = 2  # qubits
L = 1  # layers
ni = 4  # features

# linear system Ax=b
A = np.random.rand(2**n, 2**n)
A = A @ A.T
b = np.ones(2**n)

def U_b(b):
    """
    Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>.
    """
    qml.templates.MottonenStatePreparation(b / np.linalg.norm(b), wires=range(n))


def CA(idx, mats, wires):
    """
    Controlled versions of the unitary components A_l of the problem matrix A.
    """
    qml.ControlledQubitUnitary(mats[idx], control_wires=[n], wires=wires[idx])


def V(weights):
    """
    Variational circuit mapping the ground state |0> to the ansatz state |x>.

    Args:
        weights (list): Optimizable Parameters

    Returns:
        None

    """

    # for i in range(n):
    #     qml.Hadamard(wires=i)

    # for i in range(n):
    #     qml.RY(weights[i], wires=i)

    qml.BasicEntanglerLayers(weights, wires=range(n), rotation=None, id=None)


def get_paulis(A):
    """
    Decompose the input matrix into its Pauli components.

    Args:
        A (np.array): Matrix to decompose.

    Returns:
        mats (list): Pauli matrices
        wires(list): wire indices, where the Pauli matrices are applied

        O(4^n)

    """

    # decompose (complexity O(4^n))
    pauli_matrix = qml.pauli_decompose(A, check_hermitian=True, pauli=False)

    # get coefficients and operators
    coeffs = pauli_matrix.coeffs
    ops = pauli_matrix.ops

    # create Pauli word
    pw = qml.pauli.PauliWord({i:pauli for i, pauli in enumerate(ops)})

    # get wires
    wires = [pw[i].wires for i in range(len(pw))]

    # convert Pauli operator to matrix
    mats = [qml.pauli.pauli_word_to_matrix(pw[i]) for i in range(len(pw))]

    return mats, wires, pauli_matrix.coeffs

# Pauli decomposition
mats, wires, c = get_paulis(A)

def qlayer(n, l=None, lp=None, j=None, part=None):

    dev = qml.device("default.qubit.torch", wires=n+1)

    @qml.qnode(dev)
    def qcircuit(weights, inputs=None):
        """
        Variational circuit mapping the ground state |0> to the ansatz state |x>.

        Args:
            vec (np.array): Vector state to be embedded in a quantum state.

        Returns:
            Expectation value of ancilla qubit in Pauli-Z basis

        """

        # First Hadamard gate applied to the ancillary qubit.
        qml.Hadamard(wires=n)

        # For estimating the imaginary part of the coefficient "mu", we must
        # add a "-i" phase gate.
        if part == "Im" or part == "im":
            qml.PhaseShift(-np.pi / 2, wires=n)

        # Variational circuit for |x>
        V(weights)

        # Controlled application of A_l
        CA(l, mats, wires)

        # Adjoint State Preparation |b>
        qml.adjoint(U_b)(b)

        # Controlled Z operator at position j. If j = -1, apply the identity
        if j != -1:
            qml.CZ(wires=[n, j])

        # State Preparation |b>
        U_b(b)

        # Controlled application of Adjoint(A_lp)
        qml.adjoint(CA)(lp, mats, wires)

        # Second Hadamard gate applied to the ancillary qubit.
        qml.Hadamard(wires=n)

        # Expectation value of Z for the ancillary qubit.
        return qml.expval(qml.PauliZ(wires=n))

    return qcircuit

class HybridNeuralNetwork(nn.Module):

    def __init__(self, qnode, nqubits, nlayers, ninputs, npaulis):
        super().__init__()

        # number of qubits & layers & inputs & paulis
        self.nqubits = nqubits
        self.nlayers = nlayers
        self.ninputs = ninputs
        self.npaulis = npaulis

        # classical layers
        self.lin1 = nn.Linear(self.ninputs, 8)
        self.lin2 = nn.Linear(8, 8)
        self.lin3 = nn.Linear(8, self.nqubits)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # new
        self.qnode = qnode
        self.qlayers = {}
        it = 0
        for l in range(self.npaulis):
            for lp in range(self.npaulis):
                for part in ["Re", "Im"]:
                    # setattr(self, f"xx_{it}", qml.qnn.TorchLayer(self.qnode[(l, lp, -1, part)], {"weights": (self.nqubits)}))
                    setattr(self, f"xx_{it}", self.qnode[(l, lp, -1, part)])
                    self.qlayers[(l, lp, -1, part)] = getattr(self, f"xx_{it}")
                    it += 1
                    for j in range(self.nqubits):
                        # setattr(self, f"xx_{it}", qml.qnn.TorchLayer(self.qnode[(l, lp, j, part)], {"weights": (self.nqubits)}))
                        setattr(self, f"xx_{it}", self.qnode[(l, lp, j, part)])
                        self.qlayers[(l, lp, j, part)] = getattr(self, f"xx_{it}")
                        it += 1


        # something new
        init_method = functools.partial(torch.nn.init.uniform_, b=2*np.pi)
        self.alpha = torch.nn.Parameter(init_method(torch.Tensor(1, self.ninputs)), requires_grad=True)

    def forward(self, x):

        # neural network architecture
        y = self.lin1(self.alpha)
        y = self.tanh(y)
        y = self.lin2(y)
        y = self.tanh(y)
        y = self.lin3(y)
        # y = self.tanh(y)
        # y = torch.reshape(y, (self.nqubits,))
        y = torch.reshape(y, (1, self.nqubits))


        # use output of the DNN for every VQC
        outputs = {}
        for l in range(self.npaulis):
            for lp in range(self.npaulis):
                for part in ["Re", "Im"]:
                    outputs[(l, lp, -1, part)] = self.qlayers[(l, lp, -1, part)](y)
                    for j in range(self.nqubits):
                        outputs[(l, lp, j, part)] = self.qlayers[(l, lp, j, part)](y)

        return outputs, y

def create_qnodes(n, lenc, qlayer):

    qnode_dict = {}
    for l in range(lenc):
        for lp in range(lenc):
            # psi
            qn_re_psi = qlayer(n=n, l=l, lp=lp, j=-1, part="Re")
            qn_im_psi = qlayer(n=n, l=l, lp=lp, j=-1, part="Im")
            qnode_dict[(l, lp, -1, "Re")] = qn_re_psi
            qnode_dict[(l, lp, -1, "Im")] = qn_im_psi
            for j in range(n):
                # mu
                qn_re_mu = qlayer(n=n, l=l, lp=lp, j=j, part="Re")
                qn_im_mu = qlayer(n=n, l=l, lp=lp, j=j, part="Im")
                qnode_dict[(l, lp, j, "Re")] = qn_re_mu
                qnode_dict[(l, lp, j, "Im")] = qn_im_mu

    return qnode_dict

def cost(out):
    """
    Local version of the cost function. Tends to zero when A|x> is proportional
    to |b>.

    Args:
        out (np.array): outputs of all quantum circuit evaluations

    Returns:
        C_L (float): Cost function value

    """

    # Pauli decomposition
    _, _, c = get_paulis(A)

    # numpy to torch
    c = torch.from_numpy(c)

    # compute sums
    psi_norm = 0.0
    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            psi_norm += c[l] * np.conj(c[lp]) * (out[(l, lp, -1, "Re")] + 1.0j * out[(l, lp, -1, "Im")])

    # compute sums
    mu_sum = 0.0
    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            for j in range(0, n):
                mu_sum += c[l] * np.conj(c[lp]) * (out[(l, lp, j, "Re")] + 1.0j * out[(l, lp, j, "Im")])

    # Cost function
    C_L = 0.5 - 0.5 * abs(mu_sum) / (n * abs(psi_norm))

    return C_L

def optimize():

    # init hybrid model
    qlayers = create_qnodes(n, len(c), qlayer)

    model = HybridNeuralNetwork(qnode=qlayers, nqubits=n, nlayers=L, ninputs=ni, npaulis=len(c))

    maxiter = 200  # max iterations
    tol = 0.001  # threshold
    eta = 1.0  # learning rate

    # List to save data
    cost_hist = []

    # features
    x = torch.ones(ni)*(np.pi/4)

    # Optimizer
    opt = torch.optim.SGD(model.parameters(), lr=eta)

    for i in range(maxiter):

        # track time
        t0 = time()

        opt.zero_grad()  # init gradient
        out, _ = model(x)  # forward pass
        loss = cost(out)  # compute loss
        loss.backward()  # backpropagation
        opt.step()  # update weights

        # save cost function value
        cost_hist.append(loss.item())

        # print information
        if i%1 == 0:
            print("iter {:4d}    cost  {:.5f}    time  {:.4f}".format(i, loss, time()-t0))

        if loss.item() < tol:  # breaking condition
            print("\nOptimum found after {:3d} Steps!".format(i))
            _, opti_mopti = model(x)
            return cost_hist, opti_mopti, 0
    _, opti_mopti = model(x)
    return cost_hist, opti_mopti, -1

# start optimization
loss_hist, w_opt, status = optimize()

fig, axs = plt.subplots(1, 1, figsize=(6, 4))
axs.plot(loss_hist, color="purple", alpha=1.0)
axs.set_xlabel("Iteration", fontsize=15)
axs.set_ylabel("Cost", fontsize=15)
axs.spines['top'].set_linewidth(2.0)
axs.spines['bottom'].set_linewidth(2.0)
axs.spines['left'].set_linewidth(2.0)
axs.spines['right'].set_linewidth(2.0)
plt.tight_layout()
plt.show()

A_inv = np.linalg.inv(A)
x = np.dot(A_inv, b)
c_probs = (x / np.linalg.norm(x)) ** 2

n_shots = 10 ** 6

dev_x = qml.device("lightning.qubit", wires=n, shots=n_shots)

@qml.qnode(dev_x, interface="autograd")
def prepare_and_sample(weights):
    V(weights)
    return qml.sample()

# get samples
raw_samples = prepare_and_sample(w_opt)

if n == 1:
    raw_samples = [[_] for _ in raw_samples]
# convert the raw samples (bit strings) into integers and count them
samples = []
for sam in raw_samples:
    samples.append(int("".join(str(bs) for bs in sam), base=2))

# compute probabilities
q_probs = np.round(np.bincount(samples) / n_shots, 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))

ax1.bar(np.arange(0, 2 ** n), c_probs, color="red")
ax1.set_xlim(-0.5, 2 ** n - 0.5)
ax1.set_ylim(0.0, 1.0)
ax1.set_xlabel("Vector space basis")
ax1.set_title("Classical probabilities")
ax2.bar(np.arange(0, 2 ** n), q_probs, color="green")
ax2.set_xlim(-0.5, 2 ** n - 0.5)
ax2.set_ylim(0.0, 1.0)
ax2.set_xlabel("Hilbert space basis")
ax2.set_title("Quantum probabilities")
plt.show()

