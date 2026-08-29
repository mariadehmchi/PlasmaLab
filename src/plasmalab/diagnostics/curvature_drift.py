import numpy as np


def curvature_vector(magnetic_field, position, time):
    """
    Compute the curvature vector of the magnetic field line.

    kappa = (b · grad)b

    For the circular field geometry used by
    CurvedMagneticField, the curvature points
    toward the center of the circular field line.
    """

    x, y, z = position

    radius = np.sqrt(x**2 + y**2)

    if radius == 0.0:
        raise ValueError(
            "Curvature is undefined on the z-axis."
        )

    return np.array(
        [
            -x / radius**2,
            -y / radius**2,
            0.0
        ],
        dtype=float
    )


def curvature_drift_velocity(
    particle,
    magnetic_field,
    position=None,
    time=0.0
):
    """
    Compute the theoretical curvature drift velocity.

    v_curv = (m * v_parallel^2 / (q * B))
             * (b x kappa)

    Parameters
    ----------
    particle : Particle
        Charged particle.
    magnetic_field : magnetic field object
        Magnetic field model.
    position : array-like, optional
        Position at which the drift is evaluated.
        Defaults to particle.position.
    time : float
        Simulation time.
    """

    if position is None:
        position = particle.position

    B = magnetic_field.value(
        position,
        time
    )

    B_magnitude = np.linalg.norm(B)

    if B_magnitude == 0.0:
        raise ValueError(
            "Curvature drift is undefined for B = 0."
        )

    b = B / B_magnitude

    kappa = curvature_vector(
        magnetic_field,
        position,
        time
    )

    v = particle.velocity

    v_parallel = np.dot(v, b)

    return (
        particle.mass
        * v_parallel**2
        / (particle.charge * B_magnitude)
        * np.cross(b, kappa)
    )
def guiding_center_curvature_drift(
    particle,
    magnetic_field,
    positions,
    velocities,
    times
):
    """
    Estimate the curvature drift from the guiding-center motion.

    The guiding center is approximated by removing the local
    Larmor-radius contribution from the particle position.
    """

    guiding_centers = []

    for position, velocity, time in zip(
        positions,
        velocities,
        times
    ):
        B = magnetic_field.value(
            position,
            time
        )

        B_magnitude = np.linalg.norm(B)

        if B_magnitude == 0.0:
            raise ValueError(
                "Guiding center is undefined for B = 0."
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
            velocity
            - v_parallel_vector
        )

        # Local Larmor radius vector:
        #
        # rho = (m / qB) (v_perp x b)
        #
        rho = (
            particle.mass
            / (particle.charge * B_magnitude)
            * np.cross(
                v_perpendicular,
                b
            )
        )

        guiding_center = (
            position - rho
        )

        guiding_centers.append(
            guiding_center
        )

    return np.array(guiding_centers)
    