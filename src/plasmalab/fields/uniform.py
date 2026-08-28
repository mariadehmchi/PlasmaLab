import numpy as np


class UniformMagneticField:
    """
    Represents a uniform magnetic field.

    Parameters
    ----------
    field : array-like
        Magnetic field vector [Bx, By, Bz].
    """

    def __init__(self, field):

        self.field = np.array(
            field,
            dtype=float
        )

    def value(self, position, time):

        """
        Return the magnetic field at a given
        position and time.

        For a uniform magnetic field, the value
        is independent of position and time.
        """

        return self.field
class UniformElectricField:
    """
    Represents a uniform electric field.

    Parameters
    ----------
    field : array-like
        Electric field vector [Ex, Ey, Ez].
    """

    def __init__(self, field):

        self.field = np.array(
            field,
            dtype=float
        )

    def value(self, position, time):

        """
        Return the electric field at a given
        position and time.

        For a uniform electric field, the value
        is independent of position and time.
        """

        return self.field
        