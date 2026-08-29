import numpy as np

from plasmalab.particles.particle import Particle
from plasmalab.fields.curved import CurvedMagneticField
from plasmalab.fields.uniform import UniformElectricField
from plasmalab.integrators.boris import boris_step
from plasmalab.simulation.engine import Simulation


# ==================================================
# PARAMETERS
# ==================================================

B0 = 1.0
R = 10.0

mass = 1.0
charge = 1.0

initial_position = [R, 0.0, 0.0]
initial_velocity = [0.0, 1.0, 0.0]

total_time = 20.0

theoretical_drift = np.array(
    [
        0.0,
        0.0,
        1.0 / R
    ]
)


# ==================================================
# FIELDS
# ==================================================

magnetic_field = CurvedMagneticField(
    B0=B0,
    R=R
)

electric_field = UniformElectricField(
    [0.0, 0.0, 0.0]
)


# ==================================================
# TIME STEPS
# ==================================================

dt_values = [
    0.04,
    0.02,
    0.01,
    0.005
]


print("========= CURVATURE DRIFT CONVERGENCE ==========")

print()
print("Theoretical curvature drift:")
print(theoretical_drift)

print()
print(
    f"{'dt':>10} "
    f"{'measured':>15} "
    f"{'error':>15} "
    f"{'relative error':>18}"
)

print("-" * 62)


# ==================================================
# CONVERGENCE STUDY
# ==================================================

for dt in dt_values:

    steps = int(round(total_time / dt))

    particle = Particle(
        mass=mass,
        charge=charge,
        position=initial_position,
        velocity=initial_velocity
    )

    simulation = Simulation(
        particle=particle,
        electric_field=electric_field,
        magnetic_field=magnetic_field,
        integrator=boris_step,
        dt=dt,
        steps=steps
    )

    times, positions, velocities, energies = simulation.run()

    # --------------------------------------------------
    # Linear fit of z(t)
    # --------------------------------------------------

    start_index = len(times) // 2

    fit_times = times[start_index:]
    fit_z = positions[start_index:, 2]

    slope, intercept = np.polyfit(
        fit_times,
        fit_z,
        1
    )

    measured_drift = np.array(
        [
            0.0,
            0.0,
            slope
        ]
    )

    # --------------------------------------------------
    # Error
    # --------------------------------------------------

    error = np.linalg.norm(
        measured_drift
        - theoretical_drift
    )

    relative_error = (
        error
        / np.linalg.norm(theoretical_drift)
    )

    print(
        f"{dt:10.5f} "
        f"{slope:15.8f} "
        f"{error:15.8f} "
        f"{relative_error:18.8f}"
    )


print()
print("Convergence study completed.")