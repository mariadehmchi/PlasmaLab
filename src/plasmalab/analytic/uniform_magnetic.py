import numpy as np


def cyclotron_solution(
    particle,
    magnetic_field,
    times
):
    """
    Analytical solution for a charged particle
    moving in a uniform magnetic field.

    This implementation assumes:

        B = (0, 0, Bz)

    and an initial velocity perpendicular
    to the magnetic field.
    """

    # Initial conditions
    r0 = particle.position.copy()
    v0 = particle.velocity.copy()

    # Magnetic field
    B = magnetic_field.value(
        r0,
        time=0.0
    )

    Bz = B[2]

    # Cyclotron frequency
    omega = (
        particle.charge
        * Bz
        / particle.mass
    )

    # Initial perpendicular velocity
    vx0 = v0[0]
    vy0 = v0[1]

    # Exact velocity
    vx = (
        vx0 * np.cos(omega * times)
        + vy0 * np.sin(omega * times)
    )

    vy = (
        vy0 * np.cos(omega * times)
        - vx0 * np.sin(omega * times)
    )

    vz = np.full_like(
        times,
        v0[2],
        dtype=float
    )

    velocities = np.column_stack(
        (
            vx,
            vy,
            vz
        )
    )

    # Exact position
    x = (
        r0[0]
        + vx0 * np.sin(omega * times) / omega
        + vy0
        * (
            1 - np.cos(omega * times)
        )
        / omega
    )

    y = (
        r0[1]
        + vy0 * np.sin(omega * times) / omega
        - vx0
        * (
            1 - np.cos(omega * times)
        )
        / omega
    )

    z = (
        r0[2]
        + v0[2] * times
    )

    positions = np.column_stack(
        (
            x,
            y,
            z
        )
    )

    # Kinetic energy
    speed_squared = np.sum(
        velocities ** 2,
        axis=1
    )

    energies = (
        0.5
        * particle.mass
        * speed_squared
    )

    return (
        positions,
        velocities,
        energies
    )