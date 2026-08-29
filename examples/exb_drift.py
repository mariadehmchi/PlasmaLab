import numpy as np
import matplotlib.pyplot as plt
from plasmalab.particles.particle import Particle

from plasmalab.fields.uniform import (
    UniformElectricField,
    UniformMagneticField
)

from plasmalab.integrators.boris import boris_step

from plasmalab.simulation.engine import Simulation


# ==========================================
# PHYSICAL PARAMETERS
# ==========================================

mass = 1.0
charge = 1.0


# ==========================================
# INITIAL PARTICLE
# ==========================================

particle = Particle(
    mass=mass,
    charge=charge,
    position=[0.0, 0.0, 0.0],
    velocity=[1.0, 0.0, 0.0]
)


# ==========================================
# ELECTROMAGNETIC FIELDS
# ==========================================

electric_field = UniformElectricField(
    [0.2, 0.0, 0.0]
)

magnetic_field = UniformMagneticField(
    [0.0, 0.0, 1.0]
)


# ==========================================
# NUMERICAL PARAMETERS
# ==========================================

dt = 0.01

steps = 10000


# ==========================================
# RUN SIMULATION
# ==========================================

simulation = Simulation(
    particle=particle,
    electric_field=electric_field,
    magnetic_field=magnetic_field,
    integrator=boris_step,
    dt=dt,
    steps=steps
)

times, positions, velocities, energies = simulation.run()

# ==========================================
# E x B DRIFT ANALYSIS
# ==========================================

# Evaluate the fields

position = np.array([0.0, 0.0, 0.0])

E = electric_field.value(
    position,
    0.0
)

B = magnetic_field.value(
    position,
    0.0
)


# Theoretical E x B drift velocity

B_squared = np.dot(B, B)

theoretical_drift = (
    np.cross(E, B)
    / B_squared
)


# Measured drift velocity

total_time = times[-1]

measured_drift = (
    positions[-1]
    - np.array([0.0, 0.0, 0.0])
) / total_time


# Drift error

drift_error = np.linalg.norm(
    measured_drift
    - theoretical_drift
)
# ==========================================
# RESULTS
# ==========================================

print()
print("========== E x B DRIFT SIMULATION ==========")

print()

print("Electric field:", electric_field.value(
    np.array([0.0, 0.0, 0.0]),
    0.0
))

print("Magnetic field:", magnetic_field.value(
    np.array([0.0, 0.0, 0.0]),
    0.0
))

print()

print("Final position:", positions[-1])

print("Final velocity:", velocities[-1])

print("Initial kinetic energy:", energies[0])

print("Final kinetic energy:", energies[-1])
print()

print("========== E x B DRIFT ANALYSIS ==========")

print()

print("Theoretical drift velocity:")
print(theoretical_drift)

print()

print("Measured drift velocity:")
print(measured_drift)

print()

print("Drift velocity error:")
print(drift_error)
# ==========================================
# GUIDING CENTER ANALYSIS
# ==========================================

# E x B drift velocity
drift_velocity = np.cross(E, B) / np.dot(B, B)

# Velocity relative to the E x B drift
relative_velocities = velocities - drift_velocity

# Guiding center coordinates
guiding_center_x = (
    positions[:, 0]
    + relative_velocities[:, 1]
)

guiding_center_y = (
    positions[:, 1]
    - relative_velocities[:, 0]
)

# Guiding center trajectory
guiding_centers = np.column_stack(
    (
        guiding_center_x,
        guiding_center_y,
        positions[:, 2]
    )
)

# Measured guiding center velocity
guiding_center_velocity = (
    guiding_centers[-1]
    - guiding_centers[0]
) / times[-1]

guiding_center_error = np.linalg.norm(
    guiding_center_velocity
    - drift_velocity
)


print()

print("========== GUIDING CENTER ANALYSIS ==========")

print()

print("Theoretical E x B drift velocity:")
print(drift_velocity)

print()

print("Measured guiding center velocity:")
print(guiding_center_velocity)

print()

print("Guiding center drift error:")
print(guiding_center_error)
# ==========================================
# TRAJECTORY VISUALIZATION
# ==========================================

# Theoretical E x B drift trajectory
drift_position = np.zeros_like(positions)

drift_velocity = np.cross(E, B) / np.dot(B, B)

for i, t in enumerate(times):
    drift_position[i] = drift_velocity * t


plt.figure()

# Simulated particle trajectory
plt.plot(
    positions[:, 0],
    positions[:, 1],
    label="Boris simulation"
)

# Theoretical drift trajectory
plt.plot(
    drift_position[:, 0],
    drift_position[:, 1],
    "--",
    label="Theoretical E x B drift"
)

plt.xlabel("x")
plt.ylabel("y")

plt.title("Charged Particle Trajectory: E x B Drift")

plt.legend()

plt.axis("equal")

plt.grid()

plt.show()