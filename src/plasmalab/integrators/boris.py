import numpy as np


def boris_step(
    particle,
    electric_field,
    magnetic_field,
    time,
    dt
):
    """
    Advance a charged particle by one time step
    using the Boris method.
    """

    # Store the velocity at the beginning
    # of the time step
    old_velocity = particle.velocity.copy()

    # Get fields at the current particle position
    E = electric_field.value(
        particle.position,
        time
    )

    B = magnetic_field.value(
        particle.position,
        time
    )

    q_over_m = (
        particle.charge
        / particle.mass
    )

    # ------------------------------------------
    # First half acceleration by electric field
    # ------------------------------------------

    v_minus = (
        particle.velocity
        + q_over_m * E * dt / 2
    )

    # ------------------------------------------
    # Rotation in the magnetic field
    # ------------------------------------------

    t = (
        q_over_m
        * B
        * dt
        / 2
    )

    t_squared = np.dot(
        t,
        t
    )

    s = (
        2 * t
        / (1 + t_squared)
    )

    v_prime = (
        v_minus
        + np.cross(
            v_minus,
            t
        )
    )

    v_plus = (
        v_minus
        + np.cross(
            v_prime,
            s
        )
    )

    # ------------------------------------------
    # Second half acceleration by electric field
    # ------------------------------------------

    particle.velocity = (
        v_plus
        + q_over_m * E * dt / 2
    )

    # ------------------------------------------
    # Update position using average velocity
    # ------------------------------------------

    average_velocity = (
        old_velocity
        + particle.velocity
    ) / 2

    particle.position = (
        particle.position
        + average_velocity * dt
    )