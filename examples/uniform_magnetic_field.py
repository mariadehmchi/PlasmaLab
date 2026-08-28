from plasmalab.diagnostics.trajectory import (
    position_error,
    velocity_error,
    maximum_error,
    final_error
)
from plasmalab.analytic.uniform_magnetic import (
    cyclotron_solution
)
from plasmalab.diagnostics.energy import (
    relative_energy_error,
    maximum_energy_error
)
from plasmalab.simulation.engine import Simulation
from plasmalab.integrators.euler import euler_step
from plasmalab.integrators.boris import boris_step
from plasmalab.physics.lorentz import (
    lorentz_force,
    lorentz_acceleration
)
from plasmalab.particles.particle import Particle
from plasmalab.fields.uniform import (
    UniformMagneticField,
    UniformElectricField
)


# ==========================================
# PARTICLE
# ==========================================

particle = Particle(
    mass=1.0,
    charge=1.0,
    position=[0.0, 0.0, 0.0],
    velocity=[1.0, 0.0, 0.0]
)


# ==========================================
# MAGNETIC FIELD
# ==========================================

magnetic_field = UniformMagneticField(
    [0.0, 0.0, 1.0]
)


# ==========================================
# ELECTRIC FIELD
# ==========================================

electric_field = UniformElectricField(
    [0.0, 0.0, 0.0]
)


# ==========================================
# TEST FIELDS
# ==========================================

B = magnetic_field.value(
    particle.position,
    time=0.0
)

E = electric_field.value(
    particle.position,
    time=0.0
)


print("Particle mass:", particle.mass)
print("Particle charge:", particle.charge)
print("Particle position:", particle.position)
print("Particle velocity:", particle.velocity)

print()

print("Magnetic field B:", B)
print("Electric field E:", E)
# ==========================================
# LORENTZ FORCE
# ==========================================

time = 0.0

force = lorentz_force(
    particle,
    electric_field,
    magnetic_field,
    time
)

acceleration = lorentz_acceleration(
    particle,
    electric_field,
    magnetic_field,
    time
)


print()

print("Lorentz force:", force)
print("Acceleration:", acceleration)
# ==========================================
# EULER STEP TEST
# ==========================================

dt = 0.01

print()
print("========== EULER STEP ==========")

print("Initial position:", particle.position)
print("Initial velocity:", particle.velocity)

euler_step(
    particle,
    electric_field,
    magnetic_field,
    time,
    dt
)

print("New position:", particle.position)
print("New velocity:", particle.velocity)
# ==========================================
# FULL SIMULATION
# ==========================================

print()
print("========== FULL SIMULATION ==========")

# Important:
# Create a new particle because the previous
# particle was already modified by the Euler step.

simulation_particle = Particle(
    mass=1.0,
    charge=1.0,
    position=[0.0, 0.0, 0.0],
    velocity=[1.0, 0.0, 0.0]
)


dt = 0.01
steps = 1000


simulation = Simulation(
    particle=simulation_particle,
    electric_field=electric_field,
    magnetic_field=magnetic_field,
    integrator=euler_step,
    dt=dt,
    steps=steps
)


times, positions, velocities, energies = simulation.run()


print("Simulation time:", times[-1])

print("Final position:", positions[-1])

print("Final velocity:", velocities[-1])

print("Initial kinetic energy:", 0.5)

print("Final kinetic energy:", energies[-1])
# ==========================================
# BORIS SIMULATION
# ==========================================

print()
print("========== BORIS SIMULATION ==========")

# Create a fresh particle
boris_particle = Particle(
    mass=1.0,
    charge=1.0,
    position=[0.0, 0.0, 0.0],
    velocity=[1.0, 0.0, 0.0]
)

dt = 0.01
steps = 1000

boris_simulation = Simulation(
    particle=boris_particle,
    electric_field=electric_field,
    magnetic_field=magnetic_field,
    integrator=boris_step,
    dt=dt,
    steps=steps
)

times_boris, positions_boris, velocities_boris, energies_boris = (
    boris_simulation.run()
)

print("Simulation time:", times_boris[-1])
print("Final position:", positions_boris[-1])
print("Final velocity:", velocities_boris[-1])
print("Initial kinetic energy:", 0.5)
print("Final kinetic energy:", energies_boris[-1])
# ==========================================
# ENERGY DIAGNOSTICS
# ==========================================

print()
print("========== ENERGY DIAGNOSTICS ==========")

# Euler diagnostics
euler_energy_error = relative_energy_error(
    energies
)

euler_max_error = maximum_energy_error(
    energies
)

# Boris diagnostics
boris_energy_error = relative_energy_error(
    energies_boris
)

boris_max_error = maximum_energy_error(
    energies_boris
)

print()
print("Euler maximum relative energy error:")
print(euler_max_error)

print()
print("Boris maximum relative energy error:")
print(boris_max_error)
# ==========================================
# ANALYTICAL SOLUTION
# ==========================================

print()

print("========== ANALYTICAL SOLUTION ==========")

analytic_particle = Particle(
    mass=1.0,
    charge=1.0,
    position=[0.0, 0.0, 0.0],
    velocity=[1.0, 0.0, 0.0]
)

positions_exact, velocities_exact, energies_exact = (
    cyclotron_solution(
        analytic_particle,
        magnetic_field,
        times
    )
)

print("Final exact position:", positions_exact[-1])

print("Final exact velocity:", velocities_exact[-1])

print("Final exact kinetic energy:", energies_exact[-1])
# ==========================================
# TRAJECTORY ERROR ANALYSIS
# ==========================================

print()

print("========== TRAJECTORY ERROR ANALYSIS ==========")


# ------------------------------------------
# EULER ERRORS
# ------------------------------------------

euler_position_errors = position_error(
    positions,
    positions_exact
)

euler_velocity_errors = velocity_error(
    velocities,
    velocities_exact
)


# ------------------------------------------
# BORIS ERRORS
# ------------------------------------------

boris_position_errors = position_error(
    positions_boris,
    positions_exact
)

boris_velocity_errors = velocity_error(
    velocities_boris,
    velocities_exact
)


# ------------------------------------------
# RESULTS
# ------------------------------------------

print()

print("Euler final position error:")
print(final_error(euler_position_errors))

print("Euler maximum position error:")
print(maximum_error(euler_position_errors))

print()

print("Boris final position error:")
print(final_error(boris_position_errors))

print("Boris maximum position error:")
print(maximum_error(boris_position_errors))


print()

print("Euler final velocity error:")
print(final_error(euler_velocity_errors))

print("Boris final velocity error:")
print(final_error(boris_velocity_errors))