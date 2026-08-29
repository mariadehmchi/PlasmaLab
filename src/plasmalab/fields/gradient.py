import numpy as np


class GradientMagneticField:
    """
    Represents a magnetic field with a linear gradient
    along the x direction.

    The magnetic field is defined as:

        B(x) = B0 * (1 + alpha * x) * z_hat

    Parameters
    ----------
    B0 : float
        Reference magnetic field magnitude.
    alpha : float
        Relative magnetic field gradient.
    """

    def __init__(self, B0, alpha):
        self.B0 = float(B0)
        self.alpha = float(alpha)

    def value(self, position, time):
        """
        Return the magnetic field at a given
        position and time.

        The field varies with the x-coordinate
        and points in the z direction.
        """

        x = position[0]

        Bz = self.B0 * (1 + self.alpha * x)

        return np.array(
            [0.0, 0.0, Bz]
        )