import numpy as np

from plasmalab.particles.particle import Particle

from plasmalab.fields.uniform import (
    UniformElectricField,
    UniformMagneticField
)

from plasmalab.integrators.boris import boris_step


def test_boris_conserves_energy_in_magnetic_field():
    """
    In a purely magnetic field, the Boris method
    should conserve kinetic energy to numerical
    precision.
    """

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

    dt = 0.01
    steps = 1000

    # Initial kinetic energy

    initial_energy = (
        0.5
        * particle.mass
        * np.dot(
            particle.velocity,
            particle.velocity
        )
    )

    time = 0.0

    # Run the Boris integrator

    for _ in range(steps):

        boris_step(
            particle,
            electric_field,
            magnetic_field,
            time,
            dt
        )

        time += dt

    # Final kinetic energy

    final_energy = (
        0.5
        * particle.mass
        * np.dot(
            particle.velocity,
            particle.velocity
        )
    )

    assert np.isclose(
        final_energy,
        initial_energy,
        rtol=1e-12,
        atol=1e-12
    )