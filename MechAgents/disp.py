# filename: solve_displacement.py
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Plate dimensions and material properties
L = 1.0  # Length (m)
W = 1.0  # Width (m)
E = 1e9  # Young's modulus (Pa)
nu = 0.3  # Poisson's ratio

# Discretization
nx = 50  # Number of elements in x-direction
ny = 50  # Number of elements in y-direction
dx = L / nx
dy = W / ny

# Total number of nodes
N = (nx + 1) * (ny + 1)

# Assemble global stiffness matrix and force vector
K = lil_matrix((2 * N, 2 * N))
F = np.zeros(2 * N)

for j in range(ny + 1):
    for i in range(nx + 1):
        n = j * (nx + 1) + i  # Node number

        if i == nx:
            # Right edge displacement BC
            K[2 * n, 2 * n] = 1.0
            F[2 * n] = 0.1

        elif i == 0:
            # Left edge fixed BC
            K[2 * n, 2 * n] = 1.0
            K[2 * n + 1, 2 * n + 1] = 1.0


# Solve the system
U = spsolve(K.tocsr(), F)

# Reshape displacement solution
ux = U[::2].reshape(ny + 1, nx + 1)
uy = U[1::2].reshape(ny + 1, nx + 1)


# Plot the displacement field
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.linspace(0, L, nx + 1)
y = np.linspace(0, W, ny + 1)
X, Y = np.meshgrid(x, y)
ax.plot_surface(X, Y, ux, cmap='viridis')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('Displacement in x (m)')
ax.set_title('Plate Displacement')
plt.savefig('displacement.png')
print(f"Displacement solution plotted and saved to displacement.png")

