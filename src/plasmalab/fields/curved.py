import numpy as np


class CurvedMagneticField:
    """
    Toroidal magnetic field with curved field lines.

    The magnetic field follows circular field lines
    around the z-axis.

    Parameters
    ----------
    B0 : float
        Reference magnetic field magnitude.
    R : float
        Major radius of the curved field line.
    """

    def __init__(self, B0, R):
        self.B0 = float(B0)
        self.R = float(R)

    def value(self, position, time):
        """
        Return the magnetic field at a given position.

        The field direction follows the azimuthal direction
        around the z-axis.
        """

        x, y, z = position

        radius = np.sqrt(x**2 + y**2)

        if radius == 0.0:
            raise ValueError(
                "CurvedMagneticField is undefined on the z-axis."
            )

        e_phi = np.array(
            [
                -y / radius,
                x / radius,
                0.0
            ]
        )

        B_magnitude = self.B0 * self.R / radius

        return B_magnitude * e_phi