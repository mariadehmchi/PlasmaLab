from plasmalab.physics.lorentz import lorentz_acceleration


def euler_step(
    particle,
    electric_field,
    magnetic_field,
    time,
    dt
):
    """
    Advance a charged particle by one time step
    using the Euler method.

    Parameters
    ----------
    particle : Particle
        Particle to advance.

    electric_field : object
        Electric field.

    magnetic_field : object
        Magnetic field.

    time : float
        Current simulation time.

    dt : float
        Time step.
    """

    # Calculate acceleration from the Lorentz force
    acceleration = lorentz_acceleration(
        particle,
        electric_field,
        magnetic_field,
        time
    )

    # Update position using the current velocity
    particle.position = (
        particle.position
        + particle.velocity * dt
    )

    # Update velocity using the current acceleration
    particle.velocity = (
        particle.velocity
        + acceleration * dt
    )