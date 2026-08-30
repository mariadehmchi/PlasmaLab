import numpy as np

from plasmalab.particles.particle import Particle
from plasmalab.fields.mirror import MirrorMagneticField
from plasmalab.fields.uniform import UniformElectricField
from plasmalab.integrators.boris import boris_step
from plasmalab.simulation.engine import Simulation

from plasmalab.diagnostics.magnetic_mirror import (
    magnetic_moment_history
)


print("========= MAGNETIC MIRROR ADIABATIC STUDY ==========")

B0 = 1.0

alphas = [
    0.10,
    0.05,
    0.02,
    0.01
]

dt = 0.005
steps = 4000

print()
print(
    f"{'alpha':>10}"
    f"{'max mu error':>18}"
    f"{'energy error':>18}"
    f"{'turning point':>18}"
)

print("-" * 68)


for alpha in alphas:

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

    simulation = Simulation(
        particle=particle,
        electric_field=electric_field,
        magnetic_field=magnetic_field,
        integrator=boris_step,
        dt=dt,
        steps=steps
    )

    times, positions, velocities, energies = simulation.run()


    # ==================================================
    # MAGNETIC MOMENT
    # ==================================================

    moments = magnetic_moment_history(
        particle,
        magnetic_field,
        positions,
        velocities,
        times
    )

    initial_moment = moments[0]

    relative_mu_error = np.abs(
        (moments - initial_moment)
        / initial_moment
    )

    max_mu_error = np.max(
        relative_mu_error
    )


    # ==================================================
    # ENERGY
    # ==================================================

    relative_energy_error = np.abs(
        (energies - energies[0])
        / energies[0]
    )

    max_energy_error = np.max(
        relative_energy_error
    )


    # ==================================================
    # PARALLEL VELOCITY
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

        B_magnitude = np.linalg.norm(B)

        b = B / B_magnitude

        v_parallel[i] = np.dot(
            velocity,
            b
        )


    # ==================================================
    # TURNING POINT
    # ==================================================

    sign_changes = np.where(
        np.signbit(v_parallel[:-1])
        != np.signbit(v_parallel[1:])
    )[0]

    turning_point = len(sign_changes) > 0


    print(
        f"{alpha:10.3f}"
        f"{max_mu_error:18.8e}"
        f"{max_energy_error:18.8e}"
        f"{str(turning_point):>18}"
    )


print()
print("Adiabatic study completed.")