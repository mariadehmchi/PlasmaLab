from plasmalab.diagnostics.convergence import (
    observed_order
)
import numpy as np

from plasmalab.particles.particle import Particle

from plasmalab.fields.uniform import (
    UniformMagneticField,
    UniformElectricField
)

from plasmalab.integrators.euler import (
    euler_step
)

from plasmalab.integrators.boris import (
    boris_step
)

from plasmalab.simulation.engine import (
    Simulation
)

from plasmalab.analytic.uniform_magnetic import (
    cyclotron_solution
)

from plasmalab.diagnostics.trajectory import (
    position_error,
    final_error
)


# ==========================================
# PHYSICAL PARAMETERS
# ==========================================

mass = 1.0

charge = 1.0

initial_position = [
    0.0,
    0.0,
    0.0
]

initial_velocity = [
    1.0,
    0.0,
    0.0
]


# ==========================================
# FIELDS
# ==========================================

electric_field = UniformElectricField(
    [0.0, 0.0, 0.0]
)

magnetic_field = UniformMagneticField(
    [0.0, 0.0, 1.0]
)


# ==========================================
# SIMULATION PARAMETERS
# ==========================================

total_time = 10.0

time_steps = [
    0.2,
    0.1,
    0.05,
    0.025,
    0.0125,
    0.00625
]


# ==========================================
# RESULTS STORAGE
# ==========================================

euler_errors = []

boris_errors = []


# ==========================================
# CONVERGENCE STUDY
# ==========================================

for dt in time_steps:

    steps = int(
        total_time / dt
    )

    # --------------------------------------
    # EULER SIMULATION
    # --------------------------------------

    euler_particle = Particle(
        mass=mass,
        charge=charge,
        position=initial_position,
        velocity=initial_velocity
    )

    euler_simulation = Simulation(
        particle=euler_particle,
        electric_field=electric_field,
        magnetic_field=magnetic_field,
        integrator=euler_step,
        dt=dt,
        steps=steps
    )

    (
        times,
        positions_euler,
        velocities_euler,
        energies_euler
    ) = euler_simulation.run()

    # --------------------------------------
    # BORIS SIMULATION
    # --------------------------------------

    boris_particle = Particle(
        mass=mass,
        charge=charge,
        position=initial_position,
        velocity=initial_velocity
    )

    boris_simulation = Simulation(
        particle=boris_particle,
        electric_field=electric_field,
        magnetic_field=magnetic_field,
        integrator=boris_step,
        dt=dt,
        steps=steps
    )

    (
        times_boris,
        positions_boris,
        velocities_boris,
        energies_boris
    ) = boris_simulation.run()

    # --------------------------------------
    # ANALYTICAL SOLUTION
    # --------------------------------------

    exact_particle = Particle(
        mass=mass,
        charge=charge,
        position=initial_position,
        velocity=initial_velocity
    )

    (
        positions_exact,
        velocities_exact,
        energies_exact
    ) = cyclotron_solution(
        exact_particle,
        magnetic_field,
        times
    )

    # --------------------------------------
    # ERROR ANALYSIS
    # --------------------------------------

    euler_position_errors = position_error(
        positions_euler,
        positions_exact
    )

    boris_position_errors = position_error(
        positions_boris,
        positions_exact
    )

    euler_final_error = final_error(
        euler_position_errors
    )

    boris_final_error = final_error(
        boris_position_errors
    )

    euler_errors.append(
        euler_final_error
    )

    boris_errors.append(
        boris_final_error
    )

# ==========================================
# OBSERVED CONVERGENCE ORDER
# ==========================================

euler_orders = observed_order(
    time_steps,
    euler_errors
)

boris_orders = observed_order(
    time_steps,
    boris_errors
)
# ==========================================
# RESULTS
# ==========================================

print()

print("========== CONVERGENCE STUDY ==========")

print()

print(
    f"{'dt':<12}"
    f"{'Euler Error':<20}"
    f"{'Boris Error':<20}"
)

print("-" * 52)

for dt, euler_error, boris_error in zip(
    time_steps,
    euler_errors,
    boris_errors
):

    print(
        f"{dt:<12.6f}"
        f"{euler_error:<20.10e}"
        f"{boris_error:<20.10e}"
    )

print()

print("========== OBSERVED CONVERGENCE ORDER ==========")

print()

print(
    f"{'dt pair':<20}"
    f"{'Euler Order':<20}"
    f"{'Boris Order':<20}"
)

print("-" * 60)


for i in range(
    len(euler_orders)
):

    dt_1 = time_steps[i]
    dt_2 = time_steps[i + 1]

    print(
        f"{dt_1:.6f} -> {dt_2:.6f}"
        f"{'':<5}"
        f"{euler_orders[i]:<20.6f}"
        f"{boris_orders[i]:<20.6f}"
    )