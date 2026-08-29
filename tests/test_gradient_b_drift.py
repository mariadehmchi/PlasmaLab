import numpy as np

from plasmalab.particles.particle import Particle
from plasmalab.fields.gradient import GradientMagneticField
from plasmalab.fields.uniform import UniformElectricField
from plasmalab.integrators.boris import boris_step
from plasmalab.simulation.engine import Simulation

from plasmalab.diagnostics.gradient_drift import (
    gradient_b_drift_velocity
)

from plasmalab.diagnostics.gradient_guiding_center import (
    guiding_center_position
)


def test_gradient_b_drift():

    magnetic_field = GradientMagneticField(
        B0=1.0,
        alpha=0.05
    )

    electric_field = UniformElectricField(
        [0.0, 0.0, 0.0]
    )

    particle = Particle(
        mass=1.0,
        charge=1.0,
        position=[0.0, 0.0, 0.0],
        velocity=[1.0, 0.0, 0.0]
    )

    theoretical_drift = gradient_b_drift_velocity(
        particle,
        magnetic_field
    )

    simulation_particle = Particle(
        mass=1.0,
        charge=1.0,
        position=[0.0, 0.0, 0.0],
        velocity=[1.0, 0.0, 0.0]
    )

    simulation = Simulation(
        particle=simulation_particle,
        electric_field=electric_field,
        magnetic_field=magnetic_field,
        integrator=boris_step,
        dt=0.01,
        steps=1000
    )

    times, positions, velocities, energies = simulation.run()

    guiding_centers = []

    for i in range(len(times)):

        gc_particle = Particle(
            mass=1.0,
            charge=1.0,
            position=positions[i].copy(),
            velocity=velocities[i].copy()
        )

        R_gc = guiding_center_position(
            gc_particle,
            magnetic_field,
            times[i]
        )

        guiding_centers.append(R_gc)

    guiding_centers = np.array(guiding_centers)

    total_time = times[-1] - times[0]

    measured_drift = (
        guiding_centers[-1]
        - guiding_centers[0]
    ) / total_time

    error = np.linalg.norm(
        measured_drift - theoretical_drift
    )

    assert error < 0.005