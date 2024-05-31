from src.vqls.vqls_vanilla import VQLS
import pennylane.numpy as np
import matplotlib.pyplot as plt

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

x = np.linspace(-np.pi, np.pi, 5)
y = np.linspace(-np.pi, np.pi, 5)

Z = np.zeros((len(x), len(y)))
for i in range(len(x)):
    for j in range(len(y)):
        Z[i, j] = solver.cost([x[i], y[j]])

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, Z, cmap='viridis')
plt.show()

print(Z)
print("f_min", min(Z))
