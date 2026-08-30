import numpy as np

from plasmalab.particles.particle import Particle
from plasmalab.fields.mirror import MirrorMagneticField
from plasmalab.fields.uniform import UniformElectricField
from plasmalab.integrators.boris import boris_step
from plasmalab.simulation.engine import Simulation
from plasmalab.diagnostics.magnetic_mirror import (
    magnetic_moment_history
)


def run_mirror_simulation(alpha):

    magnetic_field = MirrorMagneticField(
        B0=1.0,
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
        dt=0.005,
        steps=4000
    )

    return simulation.run(), particle, magnetic_field


def test_magnetic_moment_becomes_more_adiabatic():

    results = []

    for alpha in [0.05, 0.02, 0.01]:

        (times, positions, velocities, energies), particle, field = (
            run_mirror_simulation(alpha)
        )

        moments = magnetic_moment_history(
            particle,
            field,
            positions,
            velocities,
            times
        )

        relative_error = np.max(
            np.abs(
                (moments - moments[0])
                / moments[0]
            )
        )

        results.append(relative_error)

    error_005, error_002, error_001 = results

    assert error_002 < error_005
    assert error_001 < error_002


def test_magnetic_mirror_conserves_energy():

    (times, positions, velocities, energies), _, _ = (
        run_mirror_simulation(0.01)
    )

    relative_energy_error = np.max(
        np.abs(
            (energies - energies[0])
            / energies[0]
        )
    )

    assert relative_energy_error < 1e-10


def test_magnetic_mirror_produces_turning_point():

    (times, positions, velocities, energies), particle, field = (
        run_mirror_simulation(0.01)
    )

    v_parallel = []

    for position, velocity, time in zip(
        positions,
        velocities,
        times
    ):

        B = field.value(
            position,
            time
        )

        b = B / np.linalg.norm(B)

        v_parallel.append(
            np.dot(
                velocity,
                b
            )
        )

    v_parallel = np.array(v_parallel)

    sign_changes = np.where(
        np.signbit(v_parallel[:-1])
        != np.signbit(v_parallel[1:])
    )[0]

    assert len(sign_changes) >= 1