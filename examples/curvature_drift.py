import numpy as np

from plasmalab.particles.particle import Particle
from plasmalab.fields.curved import CurvedMagneticField
from plasmalab.fields.uniform import UniformElectricField
from plasmalab.integrators.boris import boris_step
from plasmalab.simulation.engine import Simulation


print("========== CURVATURE DRIFT ==========")

B0 = 1.0
R = 10.0

magnetic_field = CurvedMagneticField(
    B0=B0,
    R=R
)

electric_field = UniformElectricField(
    [0.0, 0.0, 0.0]
)


# ==================================================
# INITIAL PARTICLE
# ==================================================

particle = Particle(
    mass=1.0,
    charge=1.0,
    position=[R, 0.0, 0.0],
    velocity=[0.0, 1.0, 0.0]
)


print("Reference magnetic field B0:", B0)
print("Radius of curvature R:", R)

print()
print("Initial position:")
print(particle.position)

print()
print("Initial velocity:")
print(particle.velocity)

print()
print("Initial magnetic field:")
print(
    magnetic_field.value(
        particle.position,
        0.0
    )
)


# ==================================================
# THEORETICAL CURVATURE DRIFT
# ==================================================

# At the initial position:
#
# b = [0, 1, 0]
#
# curvature vector:
#
# kappa = [-1/R, 0, 0]
#
# Therefore:
#
# b x kappa = [0, 0, 1/R]
#
# For m = q = v_parallel = B = 1:
#
# v_curv = 1/R
#
# so:
#
# v_curv = [0, 0, 0.1]

v_parallel = 1.0

theoretical_drift = np.array(
    [
        0.0,
        0.0,
        1.0 / R
    ]
)

print()
print("========== THEORETICAL DRIFT ==========")

print("Theoretical curvature drift velocity:")
print(theoretical_drift)


# ==================================================
# SIMULATION
# ==================================================

simulation_particle = Particle(
    mass=1.0,
    charge=1.0,
    position=[R, 0.0, 0.0],
    velocity=[0.0, 1.0, 0.0]
)

dt = 0.01
steps = 2000

simulation = Simulation(
    particle=simulation_particle,
    electric_field=electric_field,
    magnetic_field=magnetic_field,
    integrator=boris_step,
    dt=dt,
    steps=steps
)

times, positions, velocities, energies = simulation.run()


print()
print("========== CURVATURE DRIFT SIMULATION ==========")

print("Simulation time:")
print(times[-1])

print()
print("Final position:")
print(positions[-1])

print()
print("Final velocity:")
print(velocities[-1])

print()
print("Initial kinetic energy:")
print(energies[0])

print()
print("Final kinetic energy:")
print(energies[-1])


# ==================================================
# ENERGY DIAGNOSTIC
# ==================================================

relative_energy_error = np.max(
    np.abs(
        energies - energies[0]
    )
    / energies[0]
)

print()
print("========== ENERGY DIAGNOSTIC ==========")

print("Maximum relative energy error:")
print(relative_energy_error)


# ==================================================
# CURVATURE DRIFT MEASUREMENT
# ==================================================

# The particle position contains the gyromotion.
# Therefore, instead of using only:
#
#     (z_final - z_initial) / total_time
#
# we estimate the long-time drift by fitting:
#
#     z(t) = v_drift * t + intercept
#
# over the second half of the simulation.

start_index = len(times) // 2

fit_times = times[start_index:]
fit_z = positions[start_index:, 2]

slope, intercept = np.polyfit(
    fit_times,
    fit_z,
    1
)

measured_drift = np.array(
    [
        0.0,
        0.0,
        slope
    ]
)


# ==================================================
# DRIFT ERROR
# ==================================================

drift_error = np.linalg.norm(
    measured_drift
    - theoretical_drift
)

relative_drift_error = (
    drift_error
    / np.linalg.norm(theoretical_drift)
)


print()
print("========== CURVATURE DRIFT ANALYSIS ==========")

print("Fit interval:")
print(
    f"{fit_times[0]:.3f} -> {fit_times[-1]:.3f}"
)

print()
print("Measured curvature drift velocity:")
print(measured_drift)

print()
print("Theoretical curvature drift velocity:")
print(theoretical_drift)

print()
print("Curvature drift velocity error:")
print(drift_error)

print()
print("Relative curvature drift error:")
print(relative_drift_error)