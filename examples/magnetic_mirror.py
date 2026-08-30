import numpy as np

from plasmalab.particles.particle import Particle
from plasmalab.fields.mirror import MirrorMagneticField
from plasmalab.fields.uniform import UniformElectricField
from plasmalab.integrators.boris import boris_step
from plasmalab.simulation.engine import Simulation

from plasmalab.diagnostics.magnetic_mirror import (
    magnetic_moment_history
)


print("========= MAGNETIC MIRROR ==========")

B0 = 1.0
alpha = 0.1

magnetic_field = MirrorMagneticField(
    B0=B0,
    alpha=alpha
)

electric_field = UniformElectricField(
    [0.0, 0.0, 0.0]
)


particle = Particle(
    mass=1.0,
    charge=1.0,
    position=[0.0, 0.0, -2.0],
    velocity=[1.0, 0.0, 0.5]
)

print("Reference magnetic field B0:", B0)
print("Mirror parameter alpha:", alpha)

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
# SIMULATION
# ==================================================

simulation_particle = Particle(
    mass=particle.mass,
    charge=particle.charge,
    position=particle.position.copy(),
    velocity=particle.velocity.copy()
)

dt = 0.005
steps = 4000

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
print("========= MAGNETIC MIRROR SIMULATION ==========")

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

relative_energy_error = np.abs(
    (energies - energies[0])
    / energies[0]
)

print()
print("========= ENERGY DIAGNOSTIC ==========")

print("Maximum relative energy error:")
print(np.max(relative_energy_error))


# ==================================================
# PARALLEL VELOCITY ANALYSIS
# ==================================================

v_parallel = np.empty(
    len(times),
    dtype=float
)

for i, (position, velocity, time) in enumerate(
    zip(
        positions,
        velocities,
        times
    )
):

    B = magnetic_field.value(
        position,
        time
    )

    b = B / np.linalg.norm(B)

    v_parallel[i] = np.dot(
        velocity,
        b
    )


print()
print("========= PARALLEL VELOCITY ANALYSIS ==========")

print("Initial parallel velocity:")
print(v_parallel[0])

print()
print("Minimum parallel velocity:")
print(np.min(v_parallel))

print()
print("Maximum parallel velocity:")
print(np.max(v_parallel))


# ==================================================
# TURNING POINT DETECTION
# ==================================================

sign_changes = np.where(
    np.signbit(v_parallel[:-1])
    != np.signbit(v_parallel[1:])
)[0]


print()
print("========= TURNING POINT ANALYSIS ==========")

print("Number of parallel-velocity sign changes:")
print(len(sign_changes))


if len(sign_changes) > 0:

    first_turning_index = sign_changes[0]

    print()
    print("First turning point:")

    print("Time:")
    print(times[first_turning_index])

    print("Position:")
    print(positions[first_turning_index])

    print("Magnetic field:")
    print(
        magnetic_field.value(
            positions[first_turning_index],
            times[first_turning_index]
        )
    )

else:

    print()
    print("No turning point detected.")


# ==================================================
# MAGNETIC MOMENT DIAGNOSTIC
# ==================================================

moments = magnetic_moment_history(
    simulation_particle,
    magnetic_field,
    positions,
    velocities,
    times
)

initial_moment = moments[0]

relative_moment_error = np.abs(
    (moments - initial_moment)
    / initial_moment
)


print()
print("========= MAGNETIC MOMENT DIAGNOSTIC ==========")

print("Initial magnetic moment:")
print(initial_moment)

print()
print("Final magnetic moment:")
print(moments[-1])

print()
print("Maximum relative magnetic moment error:")
print(np.max(relative_moment_error))