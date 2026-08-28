import numpy as np


def lorentz_force(particle, electric_field, magnetic_field, time):
    """
    Calculate the Lorentz force acting on a charged particle.

    Parameters
    ----------
    particle : Particle
        Charged particle.
    electric_field : object
        Electric field object with a value(position, time) method.
    magnetic_field : object
        Magnetic field object with a value(position, time) method.
    time : float
        Current simulation time.

    Returns
    -------
    numpy.ndarray
        Lorentz force vector.
    """

    # Get the electric field at the particle position
    E = electric_field.value(
        particle.position,
        time
    )

    # Get the magnetic field at the particle position
    B = magnetic_field.value(
        particle.position,
        time
    )

    # Lorentz force
    F = particle.charge * (
        E + np.cross(
            particle.velocity,
            B
        )
    )

    return F
def lorentz_acceleration(
    particle,
    electric_field,
    magnetic_field,
    time
):
    """
    Calculate the acceleration of a charged particle
    using the Lorentz force.
    """

    force = lorentz_force(
        particle,
        electric_field,
        magnetic_field,
        time
    )

    acceleration = force / particle.mass

    return acceleration