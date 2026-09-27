"""
================================================================================
COMPLETE THERMO-MECHANICAL SIMULATION WITH INTEGRATED VISUALIZATIONS
================================================================================

This script runs a complete 2D transient thermo-mechanical simulation of a
304 stainless steel plate being heated by a candle. It includes:

1. Thermal analysis (heat conduction + convection)
2. Mechanical analysis (thermal stresses + deformation)
3. Complete visualizations:
   - Mesh structure
   - Temperature distribution
   - Displacement field
   - Original vs deformed shape comparison
   - Displacement magnitude
   - Time evolution plots

All results are saved as PNG images in the outputs folder.

Usage:
    python complete_simulation_corrected.py

Output:
    - 6 PNG visualization files
    - 2 time-evolution plots (temperature and displacement vs time)
    - Console output with simulation progress

================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import ufl
import pyvista as pv
from mpi4py import MPI
from dolfinx import mesh, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary, meshtags
import os

# Create output directory if it doesn't exist
output_dir = "/mnt/user-data/outputs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("\n" + "="*80)
print("THERMO-MECHANICAL SIMULATION WITH INTEGRATED VISUALIZATIONS")
print("="*80 + "\n")

# ============================================================
# 1. PARAMETERS
# ============================================================

print("[1/5] Setting up parameters...")

# Geometry
L = 1.0                  # Plate length [m]
W = 0.15                 # Plate width [m]
thickness = 0.02         # Plate thickness [m]

# Mesh
nx = 100
ny = 20

# 304 Stainless Steel
rho = 8000.0             # Density [kg/m^3]
k = 16.2                 # Thermal conductivity [W/(m K)]
cp = 500.0               # Specific heat [J/(kg K)]

E = 193e9                # Young's modulus [Pa]
nu = 0.29                # Poisson ratio
alpha = 17.3e-6          # Thermal expansion coefficient [1/K]

# Thermal conditions
T_initial = 24.0         # Initial temperature [degC]
T_ambient = 24.0         # Ambient temperature [degC]
h = 10.0                 # Convection coefficient [W/(m^2 K)]

# Candle heat source
q_candle = 50000.0       # Heat flux [W/m^2]
candle_height = 0.030    # Heated height = 20 mm
candle_y_min = (W - candle_height) / 2.0
candle_y_max = (W + candle_height) / 2.0

# Time integration
dt_value = 1.0           # Time step [s]
total_time = 600.0       # Total simulation time [s]
num_steps = int(total_time / dt_value)

# ============================================================
# 2. MPI / DOMAIN
# ============================================================

comm = MPI.COMM_WORLD
rank = comm.rank

# ============================================================
# 3. CREATE MESH
# ============================================================

print("[2/5] Creating mesh...")

domain = mesh.create_rectangle(
    comm,
    [np.array([0.0, 0.0]), np.array([L, W])],
    [nx, ny],
    cell_type=mesh.CellType.triangle,
)

tdim = domain.topology.dim
fdim = tdim - 1

# ============================================================
# 4. BOUNDARY DEFINITIONS
# ============================================================

# Candle region
facets_candle = locate_entities_boundary(
    domain,
    fdim,
    lambda x: (
        np.isclose(x[0], 0.0)
        &
        (x[1] >= candle_y_min - 1e-12)
        &
        (x[1] <= candle_y_max + 1e-12)
    ),
)

# Left edge excluding candle
facets_left = locate_entities_boundary(
    domain,
    fdim,
    lambda x: (
        np.isclose(x[0], 0.0)
        &
        (
            (x[1] < candle_y_min - 1e-12)
            |
            (x[1] > candle_y_max + 1e-12)
        )
    ),
)

# Right edge
facets_right = locate_entities_boundary(
    domain,
    fdim,
    lambda x: np.isclose(x[0], L),
)

# Bottom edge
facets_bottom = locate_entities_boundary(
    domain,
    fdim,
    lambda x: np.isclose(x[1], 0.0),
)

# Top edge
facets_top = locate_entities_boundary(
    domain,
    fdim,
    lambda x: np.isclose(x[1], W),
)

# ============================================================
# 5. CREATE FACET TAGS
# ============================================================

facet_indices = np.concatenate(
    [facets_left, facets_right, facets_bottom, facets_top, facets_candle]
)

facet_values = np.concatenate(
    [
        np.full(len(facets_left), 1, dtype=np.int32),
        np.full(len(facets_right), 2, dtype=np.int32),
        np.full(len(facets_bottom), 3, dtype=np.int32),
        np.full(len(facets_top), 4, dtype=np.int32),
        np.full(len(facets_candle), 5, dtype=np.int32),
    ]
)

sort_order = np.argsort(facet_indices)
facet_indices = facet_indices[sort_order]
facet_values = facet_values[sort_order]

facet_tags = meshtags(domain, fdim, facet_indices, facet_values)

# ============================================================
# 6. MEASURE DEFINITIONS
# ============================================================

dx = ufl.Measure("dx", domain=domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

# ============================================================
# 7. THERMAL FUNCTION SPACE & FUNCTIONS
# ============================================================

V_T = fem.functionspace(domain, ("Lagrange", 1))

T = fem.Function(V_T)
T.name = "Temperature"

T_old = fem.Function(V_T)
T_old.name = "Temperature_old"

T.x.array[:] = T_initial
T_old.x.array[:] = T_initial

# ============================================================
# 8. THERMAL VARIABLES & EQUATION
# ============================================================

T_trial = ufl.TrialFunction(V_T)
v_T = ufl.TestFunction(V_T)

# Mass/transient term
a_thermal = (rho * cp / dt_value * T_trial * v_T * thickness * dx)

# Thermal conduction
a_thermal += (k * ufl.dot(ufl.grad(T_trial), ufl.grad(v_T)) * thickness * dx)

# Edge convection
a_thermal += (h * T_trial * v_T * thickness * (ds(1) + ds(2) + ds(3) + ds(4)))

# Top and bottom surfaces
a_thermal += (2.0 * h * T_trial * v_T * dx)

# RHS
L_thermal = (rho * cp / dt_value * T_old * v_T * thickness * dx)
L_thermal += (h * T_ambient * v_T * thickness * (ds(1) + ds(2) + ds(3) + ds(4)))
L_thermal += (2.0 * h * T_ambient * v_T * dx)
L_thermal += (q_candle * thickness * v_T * ds(5))

# ============================================================
# 9. THERMAL LINEAR PROBLEM
# ============================================================

thermal_problem = LinearProblem(
    a_thermal,
    L_thermal,
    bcs=[],
    petsc_options_prefix="thermal_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)

# ============================================================
# 10. MECHANICAL FUNCTION SPACE
# ============================================================

V_u = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))

u_trial = ufl.TrialFunction(V_u)
v_u = ufl.TestFunction(V_u)
u = fem.Function(V_u)
u.name = "Displacement"

# ============================================================
# 11. PLANE-STRESS MATERIAL PARAMETERS
# ============================================================

mu = E / (2.0 * (1.0 + nu))
lambda_3d = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
lambda_ps = 2.0 * mu * lambda_3d / (lambda_3d + 2.0 * mu)

# ============================================================
# 12. STRAIN & STRESS
# ============================================================

def eps(u):
    return ufl.sym(ufl.grad(u))

def sigma_elastic(u):
    return 2.0 * mu * eps(u) + lambda_ps * ufl.tr(eps(u)) * ufl.Identity(2)

delta_T = T - T_initial
epsilon_thermal = alpha * delta_T * ufl.Identity(2)
sigma_thermal = 2.0 * mu * epsilon_thermal + lambda_ps * ufl.tr(epsilon_thermal) * ufl.Identity(2)

# ============================================================
# 13. MECHANICAL WEAK FORM
# ============================================================

a_mechanical = ufl.inner(sigma_elastic(u_trial), eps(v_u)) * dx
L_mechanical = ufl.inner(sigma_thermal, eps(v_u)) * dx

# ============================================================
# 14. FIX LEFT EDGE (DIRICHLET CONDITION)
# ============================================================

left_dofs = fem.locate_dofs_topological(V_u, fdim, facets_left)
u_zero = np.array([0.0, 0.0], dtype=np.float64)
bc_mechanical = fem.dirichletbc(u_zero, left_dofs, V_u)

# ============================================================
# 15. MECHANICAL LINEAR PROBLEM
# ============================================================

mechanical_problem = LinearProblem(
    a_mechanical,
    L_mechanical,
    bcs=[bc_mechanical],
    petsc_options_prefix="mechanical_",
    petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
)

# ============================================================
# 16. STORAGE FOR RESULTS
# ============================================================

time_values = []
max_temperature_values = []
min_temperature_values = []
max_displacement_values = []

# ============================================================
# 17. TIME LOOP
# ============================================================

if rank == 0:
    print("[3/5] Running simulation (600 time steps)...\n")
    print("="*80)
    print("TIME STEPPING")
    print("="*80)

for step in range(1, num_steps + 1):
    current_time = step * dt_value

    # THERMAL SOLUTION
    T_new = thermal_problem.solve()
    T.x.array[:] = T_new.x.array

    # MECHANICAL SOLUTION
    u_new = mechanical_problem.solve()
    u.x.array[:] = u_new.x.array

    # STORE RESULTS
    local_T_max = np.max(T.x.array)
    local_T_min = np.min(T.x.array)

    displacement_array = u.x.array.reshape((-1, 2))
    displacement_magnitude = np.sqrt(
        displacement_array[:, 0] ** 2 + displacement_array[:, 1] ** 2
    )
    local_u_max = np.max(displacement_magnitude)

    global_T_max = comm.allreduce(local_T_max, op=MPI.MAX)
    global_T_min = comm.allreduce(local_T_min, op=MPI.MIN)
    global_u_max = comm.allreduce(local_u_max, op=MPI.MAX)

    time_values.append(current_time)
    max_temperature_values.append(global_T_max)
    min_temperature_values.append(global_T_min)
    max_displacement_values.append(global_u_max)

    # UPDATE OLD TEMPERATURE
    T_old.x.array[:] = T.x.array

    # PRINT PROGRESS (every 50 steps)
    if rank == 0 and step % 50 == 0:
        print(
            f"Time = {current_time:6.1f} s | "
            f"Tmin = {global_T_min:8.2f} °C | "
            f"Tmax = {global_T_max:8.2f} °C | "
            f"Max displacement = {global_u_max * 1000.0:10.4f} mm"
        )

if rank == 0:
    print("="*80)
    print()

# ============================================================
# 18. FINAL RESULTS SUMMARY
# ============================================================

if rank == 0:
    print("="*80)
    print("SIMULATION FINISHED")
    print("="*80)
    print(f"Final minimum temperature : {min_temperature_values[-1]:.2f} °C")
    print(f"Final maximum temperature : {max_temperature_values[-1]:.2f} °C")
    print(f"Final maximum displacement: {max_displacement_values[-1] * 1000.0:.4f} mm")
    print("="*80)
    print()

# ============================================================
# VISUALIZATION FUNCTIONS
# ============================================================

def create_pyvista_grid(domain):
    """Helper function to create PyVista grid from domain - FIXED VERSION"""
    geometry = domain.geometry.x
    points = np.column_stack([
        geometry[:, 0],
        geometry[:, 1],
        np.zeros(len(geometry))
    ])

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    connectivity = domain.topology.connectivity(tdim, 0)
    cells_array = connectivity.array

    # Convert to proper VTK cell format: [3, i, j, k, 3, i, j, k, ...]
    num_cells = len(cells_array) // 3
    cells_array = cells_array.astype(np.int64)
    
    # Create the VTK cell array in proper format
    vtk_cells = []
    for i in range(0, len(cells_array), 3):
        vtk_cells.extend([3, cells_array[i], cells_array[i+1], cells_array[i+2]])
    vtk_cells = np.array(vtk_cells, dtype=np.int64)
    
    cell_types = np.full(num_cells, pv.CellType.TRIANGLE, dtype=np.uint8)

    grid = pv.UnstructuredGrid(vtk_cells, cell_types, points)
    return grid, points

# ============================================================
# VISUALIZATION 1: MESH
# ============================================================

def visualize_mesh(domain):
    """Show the mesh structure"""
    print("  → Creating mesh visualization...", end="", flush=True)
    
    grid, points = create_pyvista_grid(domain)

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(
        grid,
        color='lightblue',
        show_edges=True,
        edge_color='black',
        line_width=0.5
    )
    plotter.add_text(
        "Meshed 2D Plate Model\nBlue triangles = finite elements",
        position="upper_left",
        font_size=10
    )
    plotter.view_xy()
    plotter.window_size = (800, 600)
    plotter.screenshot(f"{output_dir}/01_mesh_structure.png")
    plotter.close()
    
    print(" ✓")

# ============================================================
# VISUALIZATION 2: TEMPERATURE
# ============================================================

def visualize_temperature(domain, T):
    """Show temperature distribution"""
    print("  → Creating temperature visualization...", end="", flush=True)
    
    grid, points = create_pyvista_grid(domain)
    temperature_values = T.x.array.copy()
    grid.point_data["Temperature"] = temperature_values

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(
        grid,
        scalars="Temperature",
        cmap="coolwarm",
        show_edges=True,
        edge_color='gray',
        scalar_bar_args={"title": "Temperature [°C]", "vertical": True}
    )
    plotter.add_text(
        "Temperature Distribution\nBlue = Cold (24°C) | Red = Hot",
        position="upper_left",
        font_size=10
    )
    plotter.view_xy()
    plotter.window_size = (800, 600)
    plotter.screenshot(f"{output_dir}/02_temperature_distribution.png")
    plotter.close()
    
    print(" ✓")

# ============================================================
# VISUALIZATION 3: DISPLACEMENT MAGNITUDE
# ============================================================

def visualize_displacement_magnitude(domain, u):
    """Show displacement magnitude"""
    print("  → Creating displacement magnitude visualization...", end="", flush=True)
    
    grid, points = create_pyvista_grid(domain)

    displacement = u.x.array.reshape((-1, 2))
    displacement_magnitude = np.sqrt(displacement[:, 0]**2 + displacement[:, 1]**2)
    grid.point_data["Displacement [mm]"] = displacement_magnitude * 1000

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(
        grid,
        scalars="Displacement [mm]",
        cmap="viridis",
        show_edges=True,
        edge_color='gray',
        scalar_bar_args={"title": "Displacement [mm]", "vertical": True}
    )
    plotter.add_text(
        "Displacement Magnitude\nBlue = Small | Yellow = Large",
        position="upper_left",
        font_size=10
    )
    plotter.view_xy()
    plotter.window_size = (800, 600)
    plotter.screenshot(f"{output_dir}/03_displacement_magnitude.png")
    plotter.close()
    
    print(" ✓")

# ============================================================
# VISUALIZATION 4: ORIGINAL VS DEFORMED
# ============================================================

def visualize_original_vs_deformed(domain, u, deformation_scale=100.0):
    """Show original vs deformed shape - FIXED VERSION"""
    print("  → Creating original vs deformed visualization...", end="", flush=True)
    
    geometry = domain.geometry.x
    original_points = np.column_stack([
        geometry[:, 0],
        geometry[:, 1],
        np.zeros(len(geometry))
    ])

    tdim = domain.topology.dim
    domain.topology.create_connectivity(tdim, 0)
    connectivity = domain.topology.connectivity(tdim, 0)
    cells_array = connectivity.array

    num_cells = len(cells_array) // 3
    cells_array = cells_array.astype(np.int64)
    
    # Create the VTK cell array in proper format
    vtk_cells = []
    for i in range(0, len(cells_array), 3):
        vtk_cells.extend([3, cells_array[i], cells_array[i+1], cells_array[i+2]])
    vtk_cells = np.array(vtk_cells, dtype=np.int64)
    
    cell_types = np.full(num_cells, pv.CellType.TRIANGLE, dtype=np.uint8)

    original_grid = pv.UnstructuredGrid(vtk_cells, cell_types, original_points)

    displacement = u.x.array.reshape((-1, 2))
    displacement_3d = np.column_stack([
        displacement[:, 0],
        displacement[:, 1],
        np.zeros(len(displacement))
    ])

    deformed_points = original_points + deformation_scale * displacement_3d
    deformed_grid = pv.UnstructuredGrid(vtk_cells, cell_types, deformed_points)

    plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(
        original_grid,
        color='white',
        show_edges=True,
        edge_color='black',
        line_width=2.0,
        opacity=0.0,
        label="Original"
    )
    plotter.add_mesh(
        deformed_grid,
        color='lightblue',
        show_edges=True,
        edge_color='darkblue',
        line_width=1.5,
        opacity=0.7,
        label="Deformed"
    )
    plotter.add_text(
        f"Original vs Deformed Shape\nDeformation exaggerated {deformation_scale:.0f}×\n"
        "Black outline = original | Blue = deformed",
        position="upper_left",
        font_size=10
    )
    plotter.view_xy()
    plotter.window_size = (800, 600)
    plotter.screenshot(f"{output_dir}/04_original_vs_deformed.png")
    plotter.close()
    
    print(" ✓")

# ============================================================
# VISUALIZATION 5: DISPLACEMENT VECTORS
# ============================================================

def visualize_displacement_vectors(domain, u):
    """Show displacement as vectors using matplotlib quiver plot"""
    print("  → Creating displacement vectors visualization...", end="", flush=True)
    
    geometry = domain.geometry.x
    displacement = u.x.array.reshape((-1, 2))
    
    # Create figure with matplotlib for quiver plot
    fig, ax = plt.subplots(figsize=(12, 3))
    
    # Sample every 5th point to avoid overcrowding
    sample_rate = 5
    x_sample = geometry[::sample_rate, 0]
    y_sample = geometry[::sample_rate, 1]
    u_sample = displacement[::sample_rate, 0] * 1000  # Convert to mm
    v_sample = displacement[::sample_rate, 1] * 1000  # Convert to mm
    
    # Calculate magnitude for color
    magnitude = np.sqrt(u_sample**2 + v_sample**2)
    
    # Create quiver plot with color based on magnitude
    quiv = ax.quiver(x_sample, y_sample, u_sample, v_sample, magnitude, 
                     cmap='hot', scale=0.1, scale_units='width')
    
    ax.set_aspect('equal')
    ax.set_xlabel('x [m]', fontsize=12)
    ax.set_ylabel('y [m]', fontsize=12)
    ax.set_title('Displacement Vectors\n(Every 5th node shown)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    cbar = plt.colorbar(quiv, ax=ax)
    cbar.set_label('Magnitude [mm]', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/05_displacement_vectors.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(" ✓")

# ============================================================
# VISUALIZATION 6 & 7: TIME EVOLUTION PLOTS
# ============================================================

def visualize_time_evolution(time_values, max_temp, min_temp, max_disp):
    """Create time evolution plots"""
    print("  → Creating time evolution plots...", end="", flush=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Temperature evolution
    ax1.plot(time_values, max_temp, 'r-', linewidth=2, label="Maximum temperature")
    ax1.plot(time_values, min_temp, 'b-', linewidth=2, label="Minimum temperature")
    ax1.set_xlabel("Time [s]", fontsize=12)
    ax1.set_ylabel("Temperature [°C]", fontsize=12)
    ax1.set_title("Temperature Evolution", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Displacement evolution
    ax2.plot(time_values, np.array(max_disp) * 1000.0, 'g-', linewidth=2)
    ax2.set_xlabel("Time [s]", fontsize=12)
    ax2.set_ylabel("Maximum displacement [mm]", fontsize=12)
    ax2.set_title("Displacement Evolution", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/06_time_evolution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(" ✓")

# ============================================================
# 19. RUN ALL VISUALIZATIONS
# ============================================================

if rank == 0:
    print("[4/5] Creating visualizations...")
    print()
    
    visualize_mesh(domain)
    visualize_temperature(domain, T)
    visualize_displacement_magnitude(domain, u)
    visualize_original_vs_deformed(domain, u, deformation_scale=100.0)
    visualize_displacement_vectors(domain, u)
    visualize_time_evolution(time_values, max_temperature_values, 
                            min_temperature_values, max_displacement_values)
    
    print()
    print("[5/5] Complete!")
    print()
    print("="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"\n✓ Simulation completed successfully!")
    print(f"\n📊 Visualizations saved to: {output_dir}/")
    print(f"\n  1. 01_mesh_structure.png")
    print(f"     → Shows the triangular mesh (4,000 elements)")
    print(f"\n  2. 02_temperature_distribution.png")
    print(f"     → Color-coded temperature field")
    print(f"     → Range: {min_temperature_values[-1]:.1f}°C to {max_temperature_values[-1]:.1f}°C")
    print(f"\n  3. 03_displacement_magnitude.png")
    print(f"     → Color-coded displacement magnitude")
    print(f"     → Maximum: {max_displacement_values[-1]*1000:.4f} mm")
    print(f"\n  4. 04_original_vs_deformed.png")
    print(f"     → Black outline (original) vs Blue (deformed)")
    print(f"     → Deformation exaggerated 100× for visibility")
    print(f"\n  5. 05_displacement_vectors.png")
    print(f"     → Red arrows showing direction and magnitude of movement")
    print(f"\n  6. 06_time_evolution.png")
    print(f"     → Temperature and displacement over 600 seconds")
    print(f"\n" + "="*80)
    print("\nAll files are saved and ready for analysis!")
    print("="*80 + "\n")

# ============================================================
# END OF PROGRAM
# ============================================================
