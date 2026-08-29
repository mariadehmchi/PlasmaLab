import numpy as np


def guiding_center_position(particle, magnetic_field, time=0.0):
    """
    Estimate the guiding-center position of a charged particle.

    R_gc = r + (v x B) / (Omega * B)

    where

        Omega = q B / m
    """

    r = particle.position
    v = particle.velocity

    B = magnetic_field.value(r, time)
    B_magnitude = np.linalg.norm(B)

    if B_magnitude == 0:
        raise ValueError(
            "Guiding center is undefined for B = 0."
        )

    omega = (
        particle.charge
        * B_magnitude
        / particle.mass
    )

    return (
        r
        + np.cross(v, B)
        / (omega * B_magnitude)
    )