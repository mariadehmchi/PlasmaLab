import numpy as np

from plasmalab.particles.particle import Particle
from plasmalab.fields.curved import CurvedMagneticField
from plasmalab.fields.uniform import UniformElectricField
from plasmalab.integrators.boris import boris_step
from plasmalab.simulation.engine import Simulation


print("========= CURVATURE DRIFT RADIUS STUDY ==========")


# ==================================================
# PHYSICAL PARAMETERS
# ==================================================

B0 = 1.0

mass = 1.0
charge = 1.0

v_parallel = 1.0
v_perpendicular = 0.0

total_time = 20.0
dt = 0.01


# ==================================================
# RADIUS VALUES
# ==================================================

radius_values = [
    10.0,
    20.0,
    50.0,
    100.0
]


# ==================================================
# ELECTRIC FIELD
# ==================================================

electric_field = UniformElectricField(
    [0.0, 0.0, 0.0]
)


# ==================================================
# STUDY
# ==================================================

print()

print(
    f"{'R':>8}"
    f"{'rho_L/R':>12}"
    f"{'theory':>12}"
    f"{'measured':>14}"
    f"{'error':>14}"
    f"{'rel. error':>14}"
)

print("-" * 76)


for R in radius_values:

    # --------------------------------------------------
    # Magnetic field
    # --------------------------------------------------

    magnetic_field = CurvedMagneticField(
        B0=B0,
        R=R
    )

    # --------------------------------------------------
    # Initial particle
    # --------------------------------------------------

    particle = Particle(
        mass=mass,
        charge=charge,
        position=[R, 0.0, 0.0],
        velocity=[0.0, v_parallel, 0.0]
    )

    # --------------------------------------------------
    # Larmor radius
    # --------------------------------------------------

    rho_L = (
        mass
        * v_parallel
        / (abs(charge) * B0)
    )

    rho_over_R = rho_L / R

    # --------------------------------------------------
    # Theoretical curvature drift
    # --------------------------------------------------

    theoretical_drift = (
        mass
        * v_parallel**2
        / (charge * B0 * R)
    )

    # --------------------------------------------------
    # Simulation
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Measure z drift
    # --------------------------------------------------

    start_index = len(times) // 2

    fit_times = times[start_index:]
    fit_z = positions[start_index:, 2]

    slope, intercept = np.polyfit(
        fit_times,
        fit_z,
        1
    )

    measured_drift = slope

    # --------------------------------------------------
    # Error
    # --------------------------------------------------

    error = abs(
        measured_drift
        - theoretical_drift
    )

    relative_error = (
        error
        / abs(theoretical_drift)
    )

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    print(
        f"{R:8.1f}"
        f"{rho_over_R:12.5f}"
        f"{theoretical_drift:12.6f}"
        f"{measured_drift:14.6f}"
        f"{error:14.6f}"
        f"{relative_error:14.6f}"
    )


print()
print("Radius study completed.")