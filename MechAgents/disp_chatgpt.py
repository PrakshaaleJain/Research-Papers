import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

# Define parameters
Lx, Ly = 1.0, 1.0  # Plate dimensions
nx, ny = 50, 50  # Grid resolution
E = 1e9  # Young's modulus (Pa)
nu = 0.3  # Poisson ratio
ux_right = 0.1  # Prescribed displacement at right edge

# Generate mesh
dx, dy = Lx / (nx - 1), Ly / (ny - 1)
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)
nodes = nx * ny

# Initialize stiffness matrix and force vector
K = lil_matrix((2 * nodes, 2 * nodes))
F = np.zeros(2 * nodes)

# Material matrix (plane stress assumption)
C = (E / (1 - nu**2)) * np.array([
    [1, nu, 0],
    [nu, 1, 0],
    [0, 0, (1 - nu) / 2]
])

# Assemble stiffness matrix (simplified for illustration purposes)
for i in range(nodes):
    K[2*i, 2*i] = 1
    K[2*i+1, 2*i+1] = 1

# Apply boundary conditions (left edge fixed, right edge displaced)
for j in range(ny):
    left_node = j * nx
    right_node = j * nx + (nx - 1)
    
    # Fix left edge (u = 0, v = 0)
    K[2*left_node, :] = 0
    K[2*left_node, 2*left_node] = 1
    K[2*left_node+1, :] = 0
    K[2*left_node+1, 2*left_node+1] = 1
    
    # Apply displacement on the right edge
    K[2*right_node, :] = 0
    K[2*right_node, 2*right_node] = 1
    F[2*right_node] = ux_right

# Solve for displacements
U = spsolve(K.tocsr(), F)
Ux = U[0::2].reshape((ny, nx))
Uy = U[1::2].reshape((ny, nx))

# Plot results
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Ux, cmap='viridis')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.set_zlabel('Displacement (m)')
ax.set_title('Displacement Field')
plt.savefig('displacement.png', dpi=300)
plt.show()
