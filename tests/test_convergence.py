import numpy as np

from plasmalab.particles.particle import Particle

from plasmalab.fields.uniform import (
    UniformElectricField,
    UniformMagneticField
)

from plasmalab.integrators.euler import euler_step
from plasmalab.integrators.boris import boris_step

from plasmalab.simulation.engine import Simulation

from plasmalab.diagnostics.convergence import (
    observed_order
)


def exact_position(time):
    """
    Exact position for:

        q = 1
        m = 1
        B = (0, 0, 1)

        r0 = (0, 0, 0)
        v0 = (1, 0, 0)
    """

    x = np.sin(time)

    y = np.cos(time) - 1

    z = 0.0

    return np.array(
        [x, y, z]
    )


def compute_position_error(
    integrator,
    dt,
    final_time=10.0
):
    """
    Run a simulation and compute the final
    position error relative to the exact solution.
    """

    steps = int(
        final_time / dt
    )

    particle = Particle(
        mass=1.0,
        charge=1.0,
        position=[0.0, 0.0, 0.0],
        velocity=[1.0, 0.0, 0.0]
    )

    electric_field = UniformElectricField(
        [0.0, 0.0, 0.0]
    )

    magnetic_field = UniformMagneticField(
        [0.0, 0.0, 1.0]
    )

    simulation = Simulation(
        particle=particle,
        electric_field=electric_field,
        magnetic_field=magnetic_field,
        integrator=integrator,
        dt=dt,
        steps=steps
    )

    times, positions, velocities, energies = (
        simulation.run()
    )

    exact = exact_position(
        times[-1]
    )

    error = np.linalg.norm(
        positions[-1] - exact
    )

    return error


def test_euler_first_order_convergence():

    time_steps = np.array(
        [0.05, 0.025, 0.0125, 0.00625]
    )

    errors = []

    for dt in time_steps:

        error = compute_position_error(
            euler_step,
            dt
        )

        errors.append(error)

    errors = np.array(errors)

    orders = observed_order(
        time_steps,
        errors
    )

    final_order = orders[-1]

    assert np.isclose(
        final_order,
        1.0,
        atol=0.05
    )


def test_boris_second_order_convergence():

    time_steps = np.array(
        [0.05, 0.025, 0.0125, 0.00625]
    )

    errors = []

    for dt in time_steps:

        error = compute_position_error(
            boris_step,
            dt
        )

        errors.append(error)

    errors = np.array(errors)

    orders = observed_order(
        time_steps,
        errors
    )

    final_order = orders[-1]

    assert np.isclose(
        final_order,
        2.0,
        atol=0.05
    )