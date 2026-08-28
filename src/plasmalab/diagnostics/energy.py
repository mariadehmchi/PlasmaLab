import numpy as np


def kinetic_energy(mass, velocities):
    """
    Calculate the kinetic energy for a sequence
    of velocity vectors.

    Parameters
    ----------
    mass : float
        Particle mass.

    velocities : numpy.ndarray
        Array of velocity vectors with shape (N, 3).

    Returns
    -------
    numpy.ndarray
        Kinetic energy at each time step.
    """

    speed_squared = np.sum(
        velocities ** 2,
        axis=1
    )

    return 0.5 * mass * speed_squared


def relative_energy_error(energies):
    """
    Calculate the relative error of the kinetic energy
    compared to the initial energy.

    Parameters
    ----------
    energies : numpy.ndarray
        Energy values during the simulation.

    Returns
    -------
    numpy.ndarray
        Relative energy error.
    """

    initial_energy = energies[0]

    return (
        energies - initial_energy
    ) / initial_energy


def maximum_energy_error(energies):
    """
    Return the maximum absolute relative energy error.
    """

    errors = relative_energy_error(
        energies
    )

    return np.max(
        np.abs(errors)
    )