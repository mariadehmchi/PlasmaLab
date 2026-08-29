import numpy as np
import matplotlib.pyplot as plt
from plasmalab.particles.particle import Particle
from plasmalab.fields.gradient import GradientMagneticField
from plasmalab.fields.uniform import UniformElectricField
from plasmalab.integrators.boris import boris_step
from plasmalab.simulation.engine import Simulation


# ==========================================
# PARTICLE
# ==========================================

particle = Particle(
    mass=1.0,
    charge=1.0,
    position=[0.0, 0.0, 0.0],
    velocity=[1.0, 0.0, 0.0]
)


# ==========================================
# MAGNETIC FIELD WITH GRADIENT
# ==========================================

magnetic_field = GradientMagneticField(
    B0=1.0,
    alpha=0.05
)


# ==========================================
# ELECTRIC FIELD
# ==========================================

electric_field = UniformElectricField(
    [0.0, 0.0, 0.0]
)


# ==========================================
# FIELD CHECK
# ==========================================

print("========== GRADIENT MAGNETIC FIELD ==========")

print("Reference magnetic field B0:", magnetic_field.B0)
print("Gradient alpha:", magnetic_field.alpha)

B0 = magnetic_field.value(
    np.array([0.0, 0.0, 0.0]),
    0.0
)

B1 = magnetic_field.value(
    np.array([1.0, 0.0, 0.0]),
    0.0
)

print("B at x = 0:", B0)
print("B at x = 1:", B1)


# ==========================================
# SIMULATION
# ==========================================

print()
print("========== GRADIENT FIELD SIMULATION ==========")

simulation_particle = Particle(
    mass=1.0,
    charge=1.0,
    position=[0.0, 0.0, 0.0],
    velocity=[1.0, 0.0, 0.0]
)

dt = 0.01
steps = 1000

simulation = Simulation(
    particle=simulation_particle,
    electric_field=electric_field,
    magnetic_field=magnetic_field,
    integrator=boris_step,
    dt=dt,
    steps=steps
)

times, positions, velocities, energies = simulation.run()


# ==========================================
# RESULTS
# ==========================================

print("Simulation time:", times[-1])

print("Initial position:", positions[0])
print("Final position:", positions[-1])

print("Initial velocity:", velocities[0])
print("Final velocity:", velocities[-1])

print("Initial kinetic energy:", energies[0])
print("Final kinetic energy:", energies[-1])
# ==========================================
# ENERGY DIAGNOSTIC
# ==========================================

initial_energy = energies[0]

relative_energy_error = np.abs(
    (energies - initial_energy)
    / initial_energy
)

print()
print("========== ENERGY DIAGNOSTIC ==========")

print(
    "Maximum relative energy error:",
    np.max(relative_energy_error)
)


# ==========================================
# SPEED DIAGNOSTIC
# ==========================================

speeds = np.linalg.norm(
    velocities,
    axis=1
)

print()
print("========== SPEED DIAGNOSTIC ==========")

print("Initial speed:", speeds[0])
print("Final speed:", speeds[-1])

print(
    "Maximum speed variation:",
    np.max(np.abs(speeds - speeds[0]))
)
# ==========================================
# TRAJECTORY PLOT
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.plot(
    positions[:, 0],
    positions[:, 1]
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Charged Particle in a Gradient Magnetic Field")

plt.axis("equal")
plt.grid(True)

plt.show()