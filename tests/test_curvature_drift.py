import numpy as np

from plasmalab.particles.particle import Particle
from plasmalab.fields.curved import CurvedMagneticField
from plasmalab.fields.uniform import UniformElectricField
from plasmalab.integrators.boris import boris_step
from plasmalab.simulation.engine import Simulation


def measure_curvature_drift(R, dt=0.01, total_time=20.0):
    """
    Measure the curvature drift velocity for a given
    magnetic-field curvature radius R.
    """

    B0 = 1.0
    mass = 1.0
    charge = 1.0

    magnetic_field = CurvedMagneticField(
        B0=B0,
        R=R
    )

    electric_field = UniformElectricField(
        [0.0, 0.0, 0.0]
    )

    particle = Particle(
        mass=mass,
        charge=charge,
        position=[R, 0.0, 0.0],
        velocity=[0.0, 1.0, 0.0]
    )

    steps = int(round(total_time / dt))

    simulation = Simulation(
        particle=particle,
        electric_field=electric_field,
        magnetic_field=magnetic_field,
        integrator=boris_step,
        dt=dt,
        steps=steps
    )

    times, positions, velocities, energies = simulation.run()

    # Use the second half of the trajectory to
    # reduce the influence of the initial transient.

    start_index = len(times) // 2

    fit_times = times[start_index:]
    fit_z = positions[start_index:, 2]

    slope, intercept = np.polyfit(
        fit_times,
        fit_z,
        1
    )

    return slope, energies


def test_curvature_drift_direction():
    """
    The curvature drift must be in the +z direction
    for the chosen particle and magnetic-field geometry.
    """

    measured_drift, energies = measure_curvature_drift(
        R=10.0
    )

    assert measured_drift > 0.0


def test_curvature_drift_magnitude():
    """
    The measured curvature drift should be reasonably
    close to the guiding-center theoretical prediction.

    For the present geometry:

        v_d = 1 / R

    For R = 10:

        v_d = 0.1
    """

    R = 10.0

    measured_drift, energies = measure_curvature_drift(
        R=R
    )

    theoretical_drift = 1.0 / R

    relative_error = abs(
        measured_drift - theoretical_drift
    ) / abs(theoretical_drift)

    # The guiding-center approximation has a finite
    # physical error for rho_L / R = 0.1.
    #
    # We therefore use a physically meaningful tolerance
    # rather than demanding machine-level agreement.

    assert relative_error < 0.10


def test_curvature_drift_scales_with_radius():
    """
    The curvature drift should approximately scale as 1/R.
    """

    drift_10, _ = measure_curvature_drift(
        R=10.0
    )

    drift_20, _ = measure_curvature_drift(
        R=20.0
    )

    ratio = drift_10 / drift_20

    # The theoretical ratio is:
    #
    # (1/10) / (1/20) = 2

    assert np.isclose(
        ratio,
        2.0,
        rtol=0.10
    )


def test_curvature_drift_conserves_energy():
    """
    In a static magnetic field with E = 0,
    the kinetic energy should remain constant.
    """

    _, energies = measure_curvature_drift(
        R=10.0
    )

    relative_energy_error = np.max(
        np.abs(
            energies - energies[0]
        )
        / energies[0]
    )

    assert relative_energy_error < 1e-12