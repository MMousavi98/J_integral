# FEniCS code Variational Fracture Mechanics
################################################################################
#
# Evaluating fracture energy predictions using phase field and gradient enhanced
# damage models for elastomers
#
# (For plane-strain cases)
#
# Authors: S.M. Mousavi, I. Ang N. Bouklas
# Email: sm2652@cornell.edu
# date: 05/01/2024
#
################################################################################
from __future__ import division
from dolfin import *
from mshr import *
import os
import sympy
import numpy as np
import matplotlib.pyplot as plt

# Parameters for Dolfin and solvers
# ----------------------------------------------------------------------------
set_log_level(LogLevel.WARNING)

# set some dolfin specific parameters
parameters["form_compiler"]["representation"]="uflacs"
parameters["form_compiler"]["optimize"]=True
parameters["form_compiler"]["cpp_optimize"]=True
parameters["form_compiler"]["quadrature_degree"]=2
info(parameters,True)

solver_u_parameters   = {"nonlinear_solver": "snes",
                         "symmetric": True,
                         "snes_solver": {"linear_solver": "mumps",
                                         "method" : "newtontr",
                                         "line_search": "cp",
                                         "preconditioner" : "hypre_amg",
                                         "maximum_iterations": 100,
                                         "absolute_tolerance": 1e-10,
                                         "relative_tolerance": 1e-10,
                                         "solution_tolerance": 1e-10,
                                         "report": True,
                                         "error_on_nonconvergence": False}}

solver_alpha_parameters = {"nonlinear_solver": "snes",
                          "symmetric": True,
                          "snes_solver": {"maximum_iterations": 300,
                                          "report": True,
                                          "linear_solver": "mumps",
                                          "method": "vinewtonssls",
                                          "absolute_tolerance": 1e-6,
                                          "relative_tolerance": 1e-6,
                                          "error_on_nonconvergence": False}}

# Element-wise projection using LocalSolver
def local_project(v, V, u=None):
    dv = TrialFunction(V)
    v_ = TestFunction(V)
    a_proj = inner(dv, v_)*dx
    b_proj = inner(v, v_)*dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    if u is None:
        u = Function(V)
        solver.solve_local_rhs(u)
        return u
    else:
        solver.solve_local_rhs(u)
        return

# Define boundary sets for boundary conditions
# ----------------------------------------------------------------------------
class bot_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0, 0.1*hsize) # near functuin takes three values: 1. given value 2. target value 3. tolerance

class top_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], H, 0.1*hsize)

class pin_point(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0], L/2, hsize) and near(x[1], 0, 0.1*hsize)

# Convert all boundary classes for visualization
bot_boundary = bot_boundary()
top_boundary = top_boundary()
pin_point = pin_point()

# Material parameters
mu    = 1                       # Shear Modulus
kappa = 1000                    # Bulk Modulus
Gc    = 0.5                     # Fracture toughness
k_ell = 1e-5                    # Residual stiffness
eta = 0                         # Viscosity coeff

# Forces
body_force = Constant((0, 0))
load_min = 0
load_max = 1
load_steps = 101

# Geometry paramaters
L, H = 1, 1              # Length (x) and height (y-direction)
resolution = 20
hsize = 0.02
ell_multi = 2
ell = Constant(ell_multi*hsize) # Length parameter

# Numerical parameters of the alternate minimization
maxiteration = 300
AM_tolerance = 1e-3

# Radius of the outer (2) and inner (1) circles for calculating J integral
r2 = 0.47
r1 = 0.4
cirle_center = 0.5

# Naming parameters for saving output
simulation_params = "Gc_%0.1f_eta_%0.1f_load_%0.1f" % (Gc, eta, load_max)
savedir   = simulation_params + "/"

# Meshing
domain = Rectangle(Point(0.0, 0.0), Point(L, H))
domain.set_subdomain(1, Circle(Point(cirle_center, 0.5*H), r2) - Circle(Point(cirle_center, 0.5*H), r1))
domain.set_subdomain(2, Rectangle(Point(0, 0.47*H), Point(L, 0.475*H)))
domain.set_subdomain(3, Rectangle(Point(0, 0.475*H), Point(L, 0.48*H)))
domain.set_subdomain(4, Rectangle(Point(0, 0.48*H), Point(L, 0.485*H)))
domain.set_subdomain(5, Rectangle(Point(0, 0.485*H), Point(L, 0.49*H)))
domain.set_subdomain(6, Rectangle(Point(0, 0.49*H), Point(L, 0.495*H)))
domain.set_subdomain(7, Rectangle(Point(0, 0.495*H), Point(L, 0.5*H)))
domain.set_subdomain(8, Rectangle(Point(0, 0.5*H), Point(L, 0.505*H)))
domain.set_subdomain(9, Rectangle(Point(0, 0.505*H), Point(L, 0.51*H)))
domain.set_subdomain(10, Rectangle(Point(0, 0.51*H), Point(L, 0.515*H)))
domain.set_subdomain(11, Rectangle(Point(0, 0.515*H), Point(L, 0.52*H)))
domain.set_subdomain(12, Rectangle(Point(0, 0.52*H), Point(L, 0.525*H)))
domain.set_subdomain(13, Rectangle(Point(0, 0.525*H), Point(L, 0.53*H)))

mesh = generate_mesh(domain, resolution)
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, mesh.domains())
top_boundary.mark(boundary_markers, 1)
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
mf = MeshFunction("size_t", mesh, 2, mesh.domains())
dx = Measure("dx", domain=mesh, subdomain_data=mf)

# Plot and save mesh
if not os.path.exists(savedir):
    os.makedirs(savedir)
plt.figure()
plot(mesh)
plt.savefig(savedir + "mesh.pdf")
plt.close()

# Define lines and points
lines = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
points = MeshFunction("size_t", mesh, mesh.topology().dim() - 2)

# show lines of interest
lines.set_all(0)
bot_boundary.mark(lines, 1)
top_boundary.mark(lines, 1)
file_results = XDMFFile(savedir + "/" + "lines.xdmf")
file_results.write(lines)

# Show points of interest
points.set_all(0)
pin_point.mark(points, 1)
file_results = XDMFFile(savedir + "/" + "points.xdmf")
file_results.write(points)

# Variational formulation
# ----------------------------------------------------------------------------
# Tensor space for projection of stress
T_DG0 = TensorFunctionSpace(mesh,'DG',0)
DG0   = FunctionSpace(mesh,'DG',0)
# Create mixed function space for elasticity
V_CG2 = VectorFunctionSpace(mesh, "Lagrange", 2)
# CG1 also defines the function space for damage
CG1 = FunctionSpace(mesh, "Lagrange", 1)
V_CG2elem = V_CG2.ufl_element()
CG1elem = CG1.ufl_element()
# Stabilized mixed FEM for incompressible elasticity
MixElem = MixedElement([V_CG2elem, CG1elem])
# Define function spaces for displacement and pressure
V = FunctionSpace(mesh, MixElem)

# Define the function, test and trial fields
w_p = Function(V)
u_p = TrialFunction(V)
v_q = TestFunction(V)
(u, p) = split(w_p)     # Displacement, pressure
(v, q) = split(v_q)   # Test functions for u, p
# Define the function, test and trial fields for damage problem
alpha  = Function(CG1)
dalpha = TrialFunction(CG1)
beta   = TestFunction(CG1)
# Define functions to save
PTensor = Function(T_DG0, name="Nominal Stress")
FTensor = Function(T_DG0, name="Deformation Gradient")
JScalar = Function(CG1, name="Volume Ratio")

# Calculating q field which is needed for finding J integral
#--------------------------------------------------------------------
q_func_space = FunctionSpace(mesh, 'CG', 1)
q_expr = Expression('sqrt(pow(x[0]-cirle_center, 2) + pow(x[1]-0.5*H, 2)) < r1 ? \
                     1.0 : (sqrt(pow(x[0]-cirle_center, 2) + pow(x[1]-0.5*H, 2)) > r2 ? 0.0 : \
                     (r2 - sqrt(pow(x[0]-cirle_center, 2) + pow(x[1]-0.5*H, 2))) / (r2 - r1))', \
                          degree=1, r1=r1, r2=r2, cirle_center=cirle_center, H=H)
q_J_integral = interpolate(q_expr, q_func_space)
q_output = File(savedir+"q_J_integral.pvd")
q_output << q_J_integral

# Dirichlet boundary condition
# --------------------------------------------------------------------
# Inclined loading
u1 = Expression([0, "(-t/L)*x[0] + t"], t=0, L=L, degree=1)
u2 = Expression([0, "(t/L)*x[0] - t"], t=0, L=L, degree=1)
bc_u1 = DirichletBC(V.sub(0), u1, top_boundary)
bc_u2 = DirichletBC(V.sub(0), u2, bot_boundary)
bc_u = [bc_u1, bc_u2]

# bc - alpha
bc_alpha0 = DirichletBC(CG1, 0, bot_boundary)
bc_alpha1 = DirichletBC(CG1, 0, top_boundary)
bc_alpha = [bc_alpha0, bc_alpha1]

# Define the energy functional of damage problem
# --------------------------------------------------------------------
# Kinematics
d = len(u)
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor
J  = det(F)
Ic = tr(C) + 1

# Define the energy functional of the elasticity problem
# ----------------------------------------------------------------------------
# Constitutive functions of the damage model
def w(alpha):           # Specific energy dissipation per unit volume
    return alpha

def a(alpha):           # Modulation function
    return (1 - alpha)**2

def b_sq(alpha):
    return (1 - alpha)**3

def P(u, alpha):        # 1st P-K tensor
    return a(alpha)*mu*(F - inv(F.T)) - b_sq(alpha)*p*J*inv(F.T)

def energy_density_function(u, alpha):
    return a(alpha)*(mu/2)*(Ic - 3 - 2*ln(J)) - b_sq(alpha)*p*(J - 1) - (1/(2*kappa))*(p**2)


elastic_energy    = ((1 - k_ell)*a(alpha)+k_ell)*(mu/2)*(Ic - 3 - 2*ln(J))*dx \
                    - b_sq(alpha)*p*(J - 1)*dx - 1/(2*kappa)*p**2*dx 

external_work     = dot(body_force, u)*dx
elastic_potential = elastic_energy - external_work

F_u = derivative(elastic_potential, w_p, v_q)
J_u = derivative(F_u, w_p, u_p)

problem_u = NonlinearVariationalProblem(F_u, w_p, bc_u, J=J_u)
solver_u  = NonlinearVariationalSolver(problem_u)
solver_u.parameters.update(solver_u_parameters)

# Define the energy functional of damage problem
# --------------------------------------------------------------------
alpha_0 = interpolate(Expression("0", degree=0), CG1)  # initial (known) alpha
dt = 1
alpha_previous = interpolate(Expression("0", degree=0), CG1)  # initial (known) alpha
z = sympy.Symbol("z", positive=True)
c_w = float(4 * sympy.integrate(sympy.sqrt(w(z)), (z, 0, 1)))

dissipated_energy = Gc/float(c_w)*(w(alpha)/ell + ell*dot(grad(alpha), grad(alpha)))*dx
damage_functional = elastic_potential + dissipated_energy
E_alpha = dt*derivative(damage_functional, alpha, beta) + eta*(alpha - alpha_previous)*beta*dx
E_alpha_alpha = derivative(E_alpha, alpha, dalpha)

# Lower and upper bounds, set to 0 and 1 respectively
alpha_lb = interpolate(Expression("x[0]>=0 & x[0]<=0.2 & near(x[1], H/2, 0.1*hsize) ? 1 : 0", \
                       hsize = hsize, L=L, H=H, degree=0), CG1)
alpha_ub = interpolate(Expression("1", degree=0), CG1)


problem_alpha = NonlinearVariationalProblem(E_alpha, alpha, bc_alpha, J=E_alpha_alpha)
problem_alpha.set_bounds(alpha_lb, alpha_ub)
solver_alpha = NonlinearVariationalSolver(problem_alpha)
solver_alpha.parameters.update(solver_alpha_parameters)

# Loading and initialization of vectors to store data of interest
load_multipliers = np.linspace(load_min, load_max, load_steps)

# Split solutions
(u, p) = w_p.split()
# Data file name
file_tot = XDMFFile(savedir + "/results.xdmf")
file_tot.parameters["rewrite_function_mesh"] = False
file_tot.parameters["functions_share_mesh"]  = True
file_tot.parameters["flush_output"]          = True

J_integral_list = []
crack_length = []
traction_x_list = []
traction_y_list = []
# Solving at each timestep
# ----------------------------------------------------------------------------
for (i_t, t) in enumerate(load_multipliers):

    print("\033[1;32m--- Starting of Time step {0:2d}: t = {1:4f} ---\033[1;m".format(i_t, t))

    iteration = 1           # Initialization of iteration loop
    err_alpha = 1         # Initialization for condition for iteration

    # Conditions for iteration
    while err_alpha > AM_tolerance and iteration < maxiteration:
        # solvers
        solver_u.solve()
        solver_alpha.solve()
        # error
        alpha_error = alpha.vector() - alpha_0.vector()
        err_alpha = alpha_error.norm('linf')
        # monitor the results
        volume_ratio = assemble(J/(L*H)*dx)
        # if MPI.rank(MPI.comm_world) == 0:
        print ("AM Iteration: {0:3d},  alpha_error: {1:>14.8f}".format(iteration, err_alpha))
        # update iteration
        alpha_0.assign(alpha)
        iteration = iteration + 1
    # updating the lower bound to account for the irreversibility
    alpha_lb.vector()[:] = alpha.vector()
    alpha_previous.assign(alpha)

    # Project
    local_project(P(u, alpha), T_DG0, PTensor)
    local_project(F, T_DG0, FTensor)
    local_project(J, CG1, JScalar)

    # Rename for paraview
    alpha.rename("Damage", "alpha")
    u.rename("Displacement", "u")
    p.rename("Pressure", "p")

    file_tot.write(alpha, t)
    file_tot.write(u, t)
    file_tot.write(p, t)
    file_tot.write(PTensor, t)
    file_tot.write(FTensor, t)
    file_tot.write(JScalar,t)

    u1.t = t
    u2.t = t

    # Post-processing
    # ----------------------------------------
    damage_values = alpha.compute_vertex_values(mesh)
    # Find the rightmost node with damage >= 0.9
    rightmost_node = None
    rightmost_x = L/5
    for vertex in vertices(mesh):
        if damage_values[vertex.index()] >= 0.9:
            if vertex.point().x() > rightmost_x:
                rightmost_vertex = vertex
                rightmost_x = vertex.point().x()

    crack_length.append(rightmost_x)
    print("\ncrack_length:", crack_length)

    # J integral calculation
    F_1 = F[0, 0]
    F_2 = F[1, 0]
    F_1_vector = as_vector([F_1, F_2])
    J_expression = -1*(energy_density_function(u, alpha)*grad(q_J_integral)[0] \
                        - inner(P(u, alpha), outer(F_1_vector, grad(q_J_integral))))
    J_integral = assemble(J_expression*dx(1))
    J_integral_list.append(J_integral)
    print("\nJ Integral List:", J_integral_list)

    # Calculating the total force on the top boundary
    P_12 = P(u, alpha)[0, 1]
    P_22 = P(u, alpha)[1, 1]
    traction_y = assemble(P_22*ds(1))
    traction_y_list.append(traction_y)
    print("\ntraction y is:", traction_y_list)
# ----------------------------------------------------------------------------
crack_length_arr = np.array(crack_length)
J_integral_arr = np.array(J_integral_list)
traction_y_arr = np.array(traction_y_list)
np.savetxt(savedir + f'/J_disp.txt', np.column_stack((crack_length_arr, J_integral_arr)), \
            header='Crack Length | J Integral List', fmt='%f', delimiter=' | ')
np.savetxt(savedir + f'/traction_disp.txt', np.column_stack((load_multipliers, traction_y_arr)), \
            header='displacament | traction', fmt='%f', delimiter=' | ')


num_plot = load_steps
plt.figure(1)
plt.plot(crack_length[1:num_plot], J_integral_list[1:num_plot], label='total')
plt.xlabel('Crack length')
plt.ylabel('J')
plt.title('J Integral')
plt.legend()
plt.savefig(savedir + '/J_Integral.pdf', transparent=True)
plt.show()

plt.figure(2)
plt.plot(load_multipliers[1:num_plot], traction_y_list[1:num_plot])
plt.xlabel('Displacement')
plt.ylabel('Total force')
plt.title('Traction')
plt.savefig(savedir + '/traction_disp.pdf', transparent=True)

