import numpy as np


def observed_order(
    time_steps,
    errors
):
    """
    Calculate the observed order of convergence.

    The error is assumed to behave approximately as:

        error = C * dt^p

    where p is the order of convergence.

    Parameters
    ----------
    time_steps : array-like
        Sequence of time-step sizes.

    errors : array-like
        Corresponding numerical errors.

    Returns
    -------
    numpy.ndarray
        Observed convergence order between
        consecutive time steps.
    """

    time_steps = np.asarray(
        time_steps,
        dtype=float
    )

    errors = np.asarray(
        errors,
        dtype=float
    )

    return np.log(
        errors[:-1] / errors[1:]
    ) / np.log(
        time_steps[:-1] / time_steps[1:]
    )