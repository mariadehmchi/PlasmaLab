import numpy as np


def magnetic_moment(
    particle,
    magnetic_field,
    position,
    velocity,
    time
):
    """
    Compute the first adiabatic invariant:

        mu = m * v_perp^2 / (2 * B)

    where v_perp is the velocity component perpendicular
    to the local magnetic field.
    """

    B = magnetic_field.value(
        position,
        time
    )

    B_magnitude = np.linalg.norm(B)

    if B_magnitude == 0.0:
        raise ValueError(
            "Magnetic field magnitude cannot be zero."
        )

    b = B / B_magnitude

    v_parallel = np.dot(
        velocity,
        b
    )

    v_parallel_vector = (
        v_parallel * b
    )

    v_perpendicular = (
        velocity - v_parallel_vector
    )

    v_perpendicular_squared = np.dot(
        v_perpendicular,
        v_perpendicular
    )

    return (
        particle.mass
        * v_perpendicular_squared
        / (2.0 * B_magnitude)
    )


def magnetic_moment_history(
    particle,
    magnetic_field,
    positions,
    velocities,
    times
):
    """
    Compute the magnetic moment at every simulation step.
    """

    moments = np.empty(
        len(times),
        dtype=float
    )

    for i, (
        position,
        velocity,
        time
    ) in enumerate(
        zip(
            positions,
            velocities,
            times
        )
    ):
        moments[i] = magnetic_moment(
            particle,
            magnetic_field,
            position,
            velocity,
            time
        )

    return moments