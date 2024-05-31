import unittest
import numpy as np
from src.vqls.vqls_vanilla import VQLS


class TestQuantumSolver(unittest.TestCase):

    def test_circuit_construction(self):
        A = np.eye(2)
        b = np.ones(2)
        solver = VQLS(A, b)

        self.assertEqual(2 ** solver.nqubits, len(b))
        self.assertTrue(solver.nqubits > 0)


if __name__ == '__main__':
    unittest.main()
