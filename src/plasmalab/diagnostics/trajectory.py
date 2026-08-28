import numpy as np


def position_error(
    numerical_positions,
    exact_positions
):
    """
    Calculate the Euclidean position error
    at every simulation time step.

    Parameters
    ----------
    numerical_positions : numpy.ndarray
        Numerical particle positions.

    exact_positions : numpy.ndarray
        Exact particle positions.

    Returns
    -------
    numpy.ndarray
        Position error at each time step.
    """

    difference = (
        numerical_positions
        - exact_positions
    )

    return np.linalg.norm(
        difference,
        axis=1
    )


def velocity_error(
    numerical_velocities,
    exact_velocities
):
    """
    Calculate the Euclidean velocity error
    at every simulation time step.
    """

    difference = (
        numerical_velocities
        - exact_velocities
    )

    return np.linalg.norm(
        difference,
        axis=1
    )


def maximum_error(errors):
    """
    Return the maximum absolute error.
    """

    return np.max(
        np.abs(errors)
    )


def final_error(errors):
    """
    Return the final error.
    """

    return errors[-1]