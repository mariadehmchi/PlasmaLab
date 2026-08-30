import numpy as np


class MirrorMagneticField:
    """
    Divergence-free magnetic mirror field.

    The magnetic-field magnitude on the symmetry axis is

        B_z(z) = B0 * (1 + alpha * z^2)

    The transverse components are constructed so that

        div(B) = 0.

    Parameters
    ----------
    B0 : float
        Magnetic-field magnitude at z = 0.

    alpha : float
        Controls the strength of the mirror.
    """

    def __init__(self, B0, alpha):
        self.B0 = float(B0)
        self.alpha = float(alpha)

    def value(self, position, time):
        """
        Return the magnetic field at a given position and time.
        """

        position = np.asarray(position, dtype=float)

        x = position[0]
        y = position[1]
        z = position[2]

        Bz = self.B0 * (1.0 + self.alpha * z**2)

        dBz_dz = 2.0 * self.B0 * self.alpha * z

        Bx = -0.5 * dBz_dz * x
        By = -0.5 * dBz_dz * y

        return np.array(
            [
                Bx,
                By,
                Bz
            ]
        )