from fenics import *

# Define geometry and mesh
L = 1.0  # Length of the plate
mesh = UnitSquareMesh(100, 100)

# Define function space
V = VectorFunctionSpace(mesh, "Lagrange", 2)

# Define boundary conditions
left =  CompiledSubDomain("near(x[0], 0.0)")
right = CompiledSubDomain("near(x[0], 1.0)")

u_left = Constant((0.0, 0.0))
u_right = Constant((0.1, 0.0))

bc_left = DirichletBC(V, u_left, left)
bc_right = DirichletBC(V, u_right, right)
bcs = [bc_left, bc_right]

# Define strain and stress
def epsilon(u):
    return 0.5 * (grad(u) + grad(u).T)

def sigma(u):
    E = 1e9 # Young's modulus
    nu = 0.3 # Poisson's ratio
    lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    return lambda_ * tr(epsilon(u)) * Identity(2) + 2 * mu * epsilon(u)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant((0.0, 0.0)) # No body force
a = inner(sigma(u), epsilon(v)) * dx
L = dot(f, v) * dx

# Solve
u = Function(V)
solve(a == L, u, bcs)

# Save solution in VTK format
vtkfile = File('displacement.pvd')
vtkfile << u

# Plot and save the x-displacement
plot(u[0])
import matplotlib.pyplot as plt
plt.savefig("displacement.png")

print("Displacement solution saved to displacement.pvd and displacement.png")