import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
from time import time

n_qubits = 2
A = np.eye(2 ** n_qubits, 2 ** n_qubits)
A[0, 0] = 2.0
b = np.ones(2 ** n_qubits)
b = b / np.linalg.norm(b)

tot_qubits = n_qubits + 1
ancilla_idx = n_qubits


def U_b(vec):
    """
    Unitary matrix rotating the ground state to the problem vector |b> = U_b |0>.
    """
    qml.AmplitudeEmbedding(features=vec, wires=range(n_qubits), normalize=True)  # O(n^2)


def CA(idx, matrices, qubits):
    """
    Controlled versions of the unitary components A_l of the problem matrix A.
    """
    qml.ControlledQubitUnitary(matrices[idx], control_wires=[ancilla_idx], wires=qubits[idx])


def V(weights):
    """
    Variational circuit mapping the ground state |0> to the ansatz state |x>.
    """

    for idx in range(n_qubits):
        qml.Hadamard(wires=idx)

    for idx, element in enumerate(weights):
        qml.RY(element, wires=idx)


def get_paulis(mat):
    """
  Decompose the input matrix into its Pauli components in O(4^n) time

  Args:
      mat (np.array): Matrix to decompose.

  Returns:
      mats (list): Pauli matrices
      wires(list): wire indices, where the Pauli matrices are applied

  """

    # decompose
    pauli_matrix = qml.pauli_decompose(mat, check_hermitian=True, pauli=False)

    # get coefficients and operators
    coeffs = pauli_matrix.coeffs
    ops = pauli_matrix.ops

    # create Pauli word
    pw = qml.pauli.PauliWord({i: pauli for i, pauli in enumerate(ops)})

    # get wires
    qubits = [pw[i].wires for i in range(len(pw))]

    # convert Pauli operator to matrix
    matrices = [qml.pauli.pauli_word_to_matrix(pw[i]) for i in range(len(pw))]

    return matrices, qubits, coeffs


mats, wires, c = get_paulis(A)

dev_mu = qml.device("default.qubit", wires=tot_qubits)


@qml.qnode(dev_mu)
def qcircuit(weights, l=None, lp=None, j=None, part=None):
    """
  Variational circuit mapping the ground state |0> to the ansatz state |x>.

  Args:
      vec (np.array): Vector state to be embedded in a quantum state.

  Returns:
      Expectation value of ancilla qubit in Pauli-Z basis
      :param weights:
      :param l:
      :param lp:
      :param j:
      :param part:

  """

    # First Hadamard gate applied to the ancillary qubit.
    qml.Hadamard(wires=ancilla_idx)

    # For estimating the imaginary part of the coefficient "mu", we must add a "-i"
    # phase gate.
    if part == "Im" or part == "im":
        qml.PhaseShift(-np.pi / 2, wires=ancilla_idx)

    # Variational circuit generating a guess for the solution vector |x>
    V(weights)

    # Controlled application of the unitary component A_l of the problem matrix A.
    CA(l, mats, wires)

    # Adjoint of the unitary U_b associated to the problem vector |b>.
    qml.adjoint(U_b)(b)

    # Controlled Z operator at position j. If j = -1, apply the identity.
    if j != -1:
        qml.CZ(wires=[ancilla_idx, j])

    # Unitary U_b associated to the problem vector |b>.
    U_b(b)

    # Controlled application of Adjoint(A_lp).
    qml.adjoint(CA)(lp, mats, wires)

    # Second Hadamard gate applied to the ancillary qubit.
    qml.Hadamard(wires=ancilla_idx)

    # Expectation value of Z for the ancillary qubit.
    return qml.expval(qml.PauliZ(wires=ancilla_idx))


def cost(weights):
    """
  Local version of the cost function. Tends to zero when A|x> is proportional to |b>.

  Args:
      weights (np.array): trainable parameters for the variational circuit.

  Returns:
      Cost function value (float)

  """

    mu_sum = 0.0
    psi_norm = 0.0
    for l in range(0, len(c)):
        for lp in range(0, len(c)):
            psi_real = qcircuit(weights, l=l, lp=lp, j=-1, part="Re")
            psi_imag = qcircuit(weights, l=l, lp=lp, j=-1, part="Im")
            psi_norm += c[l] * np.conj(c[lp]) * (psi_real + 1.0j * psi_imag)
            for j in range(0, n_qubits):
                mu_real = qcircuit(weights, l=l, lp=lp, j=j, part="Re")
                mu_imag = qcircuit(weights, l=l, lp=lp, j=j, part="Im")
                mu_sum += c[l] * np.conj(c[lp]) * (mu_real + 1.0j * mu_imag)

    # Cost function C_L
    return 0.5 - 0.5 * abs(mu_sum) / (n_qubits * abs(psi_norm))


# optimization configs
steps = 20  # Number of optimization steps
eta = 0.8  # Learning rate
q_delta = 0.001 * np.pi  # Initial spread of random quantum weights
np.random.seed(0)
layers = 1

# initial weights for strongly entangling layers
w = q_delta * np.random.randn(n_qubits, requires_grad=True)

# Gradient Descent Optimization Algorithm
opt = qml.GradientDescentOptimizer(eta)

# Optimization loop
cost_history = []
t0 = time()
for it in range(steps):
    ta = time()
    w, cost_val = opt.step_and_cost(cost, w)
    print("Step {:3d}    obj = {:9.7f}    time = {:9.7f} sec".format(it, cost_val, time() - ta))
    if np.abs(cost_val) < 1e-4:
        break
    cost_history.append(cost_val)
print("\n Total Optimization Time: ", time() - t0, " sec")

plt.figure(1)
plt.plot(cost_history, "k", linewidth=2)
plt.ylabel("Cost function")
plt.xlabel("Optimization steps")

# classical probabilities
A_inv = np.linalg.inv(A)
x = np.dot(A_inv, b)
c_probs = (x / np.linalg.norm(x)) ** 2

# quantum probabilities
n_shots = 10 ** 6
dev_x = qml.device("lightning.qubit", wires=n_qubits, shots=n_shots)


@qml.qnode(dev_x, interface="autograd")
def prepare_and_sample(weights):
    V(weights)
    return qml.sample()


raw_samples = prepare_and_sample(w)
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
    V(weights)  # V(weight)|0>
    return qml.state()  # |x>


print(" x  =", np.round(x / np.linalg.norm(x), 2))
print("|x> =", np.round(np.real(prepare_and_get_state(w)), 2))
