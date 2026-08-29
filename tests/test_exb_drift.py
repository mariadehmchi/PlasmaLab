import numpy as np

from plasmalab.particles.particle import Particle
from plasmalab.fields.uniform import (
    UniformElectricField,
    UniformMagneticField
)
from plasmalab.integrators.boris import boris_step


def test_guiding_center_exb_drift():

    particle = Particle(
        mass=1.0,
        charge=1.0,
        position=np.array([0.0, 0.0, 0.0]),
        velocity=np.array([1.0, 0.0, 0.0])
    )

    electric_field = UniformElectricField(
        np.array([0.2, 0.0, 0.0])
    )

    magnetic_field = UniformMagneticField(
        np.array([0.0, 0.0, 1.0])
    )

    dt = 0.01
    steps = 10000

    positions = []
    velocities = []

    for i in range(steps):

        positions.append(
            particle.position.copy()
        )

        velocities.append(
            particle.velocity.copy()
        )

        boris_step(
            particle,
            electric_field,
            magnetic_field,
            i * dt,
            dt
        )

    positions = np.array(positions)
    velocities = np.array(velocities)

    # Theoretical E x B drift velocity

    E = electric_field.field
    B = magnetic_field.field

    theoretical_velocity = (
        np.cross(E, B)
        / np.dot(B, B)
    )

    # Velocity relative to the drift

    relative_velocities = (
        velocities - theoretical_velocity
    )

    # Guiding center coordinates

    guiding_center_x = (
        positions[:, 0]
        + relative_velocities[:, 1]
    )

    guiding_center_y = (
        positions[:, 1]
        - relative_velocities[:, 0]
    )

    guiding_centers = np.column_stack(
        (
            guiding_center_x,
            guiding_center_y,
            positions[:, 2]
        )
    )

    # Measured guiding center velocity

    measured_velocity = (
        guiding_centers[-1]
        - guiding_centers[0]
    ) / ((steps - 1) * dt)

    error = np.linalg.norm(
        measured_velocity
        - theoretical_velocity
    )

    assert error < 1e-10