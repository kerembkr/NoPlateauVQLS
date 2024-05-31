import pennylane as qml
import pennylane.numpy as np


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
