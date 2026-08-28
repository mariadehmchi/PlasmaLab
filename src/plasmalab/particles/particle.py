import numpy as np


class Particle:
    """
    Represents a charged particle.

    Parameters
    ----------
    mass : float
        Mass of the particle.
    charge : float
        Electric charge of the particle.
    position : array-like
        Initial position [x, y, z].
    velocity : array-like
        Initial velocity [vx, vy, vz].
    """

    def __init__(self, mass, charge, position, velocity):

        self.mass = float(mass)
        self.charge = float(charge)

        self.position = np.array(
            position,
            dtype=float
        )

        self.velocity = np.array(
            velocity,
            dtype=float
        )