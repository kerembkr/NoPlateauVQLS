import pennylane as qml
import pennylane.numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os

def init_params():
    raise NotImplementedError("to be implemented")


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


def get_random_ls(nqubits, easy_example=False):

    if easy_example:
        A_ = np.eye(2 ** nqubits)
        A_[0, 0] = 2.0
        b_ = np.ones(2 ** nqubits)
        return A_, b_

    M = np.random.rand(2 ** nqubits, 2 ** nqubits)
    A_ = M @ M.T
    # vector
    b_ = np.random.rand(2 ** nqubits)
    b_ = b_ / np.linalg.norm(b_)

    return A_, b_


def plot_costs(data, save_png=False, title=None):

    # plot curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for label, cost in data.items():
        ax.plot(cost, linewidth=2.0, label=label)
    ax.set_xlabel("Number of Iterations", fontsize=18, labelpad=15, fontname='serif')
    ax.set_ylabel("Cost Function Value", fontsize=18, labelpad=15,  fontname='serif')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(direction="in", labelsize=12, length=10, width=0.8, colors='k')
    ax.spines['top'].set_linewidth(2.0)
    ax.spines['bottom'].set_linewidth(2.0)
    ax.spines['left'].set_linewidth(2.0)
    ax.spines['right'].set_linewidth(2.0)
    ax.legend()
    legend = ax.legend(frameon=True, fontsize=12)
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(1.2)
    if title is not None:
        ax.set_title(title, fontsize=18, fontname='serif')

    if save_png:
        output_dir = '../../output/'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(os.path.join(output_dir, 'curves.png'))
