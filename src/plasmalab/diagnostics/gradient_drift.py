import numpy as np


def gradient_b_drift_velocity(
    particle,
    magnetic_field,
    time=0.0
):
    """
    Compute the theoretical gradient-B drift velocity.

    v_gradB = (m v_perp^2 / (2 q B^3)) (B x grad(B))

    This implementation assumes a magnetic field along z
    and a gradient along x.
    """

    position = particle.position
    velocity = particle.velocity

    B_vector = magnetic_field.value(position, time)
    B_magnitude = np.linalg.norm(B_vector)

    if B_magnitude == 0:
        raise ValueError(
            "Gradient-B drift is undefined for B = 0."
        )

    # Perpendicular velocity
    B_unit = B_vector / B_magnitude

    v_parallel = np.dot(velocity, B_unit) * B_unit
    v_perp = velocity - v_parallel

    v_perp_squared = np.dot(v_perp, v_perp)

    # Numerical gradient of |B|
    dx = 1e-6

    position_plus = position.copy()
    position_minus = position.copy()

    position_plus[0] += dx
    position_minus[0] -= dx

    B_plus = np.linalg.norm(
        magnetic_field.value(position_plus, time)
    )

    B_minus = np.linalg.norm(
        magnetic_field.value(position_minus, time)
    )

    grad_B = np.array([
        (B_plus - B_minus) / (2 * dx),
        0.0,
        0.0
    ])

    drift = (
        particle.mass
        * v_perp_squared
        / (2 * particle.charge * B_magnitude**3)
        * np.cross(B_vector, grad_B)
    )

    return drift