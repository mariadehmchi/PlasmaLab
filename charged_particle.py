import numpy as np
import matplotlib.pyplot as plt


# ==================================================
# EULER METHOD
# ==================================================

def simulate_euler(m, q, r0, v0, B, dt, steps):
    """
    Simulate the motion of a charged particle
    in a uniform magnetic field using Euler's method.
    """

    # Create independent copies of the initial conditions
    r = r0.copy()
    v = v0.copy()

    # Store simulation data
    positions = []
    velocities = []
    energies = []

    # Time evolution
    for i in range(steps):

        # Lorentz force
        F = q * np.cross(v, B)

        # Acceleration
        a = F / m

        # Update velocity
        v = v + a * dt

        # Update position
        r = r + v * dt

        # Store results
        positions.append(r.copy())
        velocities.append(v.copy())

        # Kinetic energy
        energy = 0.5 * m * np.dot(v, v)
        energies.append(energy)

    return (
        np.array(positions),
        np.array(velocities),
        np.array(energies)
    )


# ==================================================
# BORIS METHOD
# ==================================================

def simulate_boris(m, q, r0, v0, B, dt, steps):
    """
    Simulate the motion of a charged particle
    in a uniform magnetic field using the Boris method.
    """

    # Create independent copies of the initial conditions
    r = r0.copy()
    v = v0.copy()

    # Store simulation data
    positions = []
    velocities = []
    energies = []

    # Boris parameters
    t = (q * B / m) * (dt / 2)

    t_squared = np.dot(t, t)

    s = 2 * t / (1 + t_squared)

    # Time evolution
    for i in range(steps):

        # First Boris rotation
        v_prime = v + np.cross(v, t)

        # Second Boris rotation
        v = v + np.cross(v_prime, s)

        # Update position
        r = r + v * dt

        # Store results
        positions.append(r.copy())
        velocities.append(v.copy())

        # Kinetic energy
        energy = 0.5 * m * np.dot(v, v)
        energies.append(energy)

    return (
        np.array(positions),
        np.array(velocities),
        np.array(energies)
    )

# ==================================================
# ANALYTICAL SOLUTION
# ==================================================

def analytical_solution(m, q, r0, v0, B, dt, steps):
    """
    Analytical solution for a charged particle
    in a uniform magnetic field B = (0, 0, Bz).

    This implementation assumes the initial velocity
    is perpendicular to the magnetic field.
    """

    time = np.arange(1, steps + 1) * dt

    # Cyclotron frequency
    omega = q * B[2] / m

    # Exact position
    x = np.sin(omega * time) / omega
    y = (np.cos(omega * time) - 1) / omega
    z = np.zeros_like(time)

    positions = np.column_stack((x, y, z))

    # Exact velocity
    vx = np.cos(omega * time)
    vy = -np.sin(omega * time)
    vz = np.zeros_like(time)

    velocities = np.column_stack((vx, vy, vz))

    # Exact kinetic energy
    speed_squared = vx**2 + vy**2 + vz**2
    energies = 0.5 * m * speed_squared

    return positions, velocities, energies
# ==================================================
# PHYSICAL PARAMETERS
# ==================================================

m = 1.0
q = 1.0

# Initial position
r0 = np.array([0.0, 0.0, 0.0])

# Initial velocity
v0 = np.array([1.0, 0.0, 0.0])

# Uniform magnetic field
B = np.array([0.0, 0.0, 1.0])


# ==================================================
# NUMERICAL PARAMETERS
# ==================================================

dt = 0.01
steps = 1000


# ==================================================
# RUN SIMULATIONS
# ==================================================

# Euler simulation
positions_euler, velocities_euler, energies_euler = simulate_euler(
    m, q, r0, v0, B, dt, steps
)

# Boris simulation
positions_boris, velocities_boris, energies_boris = simulate_boris(
    m, q, r0, v0, B, dt, steps
)

# Analytical solution
positions_exact, velocities_exact, energies_exact = analytical_solution(
    m, q, r0, v0, B, dt, steps
)
# ==================================================
# ANALYSIS
# ==================================================

time = np.arange(1, steps + 1) * dt

initial_energy = 0.5 * m * np.dot(v0, v0)


print("\n========== EULER METHOD ==========")

print("Final position:", positions_euler[-1])
print("Final velocity:", velocities_euler[-1])
print("Initial kinetic energy:", initial_energy)
print("Final kinetic energy:", energies_euler[-1])


print("\n========== BORIS METHOD ==========")

print("Final position:", positions_boris[-1])
print("Final velocity:", velocities_boris[-1])
print("Initial kinetic energy:", initial_energy)
print("Final kinetic energy:", energies_boris[-1])
# ==================================================
# NUMERICAL ERROR ANALYSIS
# ==================================================

# Position error for Euler
error_euler = np.linalg.norm(
    positions_euler - positions_exact,
    axis=1
)

# Position error for Boris
error_boris = np.linalg.norm(
    positions_boris - positions_exact,
    axis=1
)

print("\n========== NUMERICAL ERROR ==========")

print("Final Euler position error:", error_euler[-1])
print("Final Boris position error:", error_boris[-1])


# ==================================================
# PLOT TRAJECTORY COMPARISON
# ==================================================

plt.figure()

plt.plot(
    positions_euler[:, 0],
    positions_euler[:, 1],
    label="Euler"
)

plt.plot(
    positions_boris[:, 0],
    positions_boris[:, 1],
    label="Boris"
)
plt.plot(
    positions_exact[:, 0],
    positions_exact[:, 1],
    label="Analytical solution",
    linestyle="--"
)
plt.xlabel("x")
plt.ylabel("y")

plt.title("Charged Particle Trajectory: Euler vs Boris")

plt.legend()

plt.axis("equal")
plt.grid()

plt.show()


# ==================================================
# PLOT ENERGY COMPARISON
# ==================================================

plt.figure()

plt.plot(
    time,
    energies_euler,
    label="Euler"
)

plt.plot(
    time,
    energies_boris,
    label="Boris"
)
plt.plot(
    time,
    energies_exact,
    label="Exact",
    linestyle="--"
)
plt.xlabel("Time")
plt.ylabel("Kinetic Energy")

plt.title("Kinetic Energy: Euler vs Boris")

plt.legend()
plt.grid()

plt.show()
# ==================================================
# PLOT NUMERICAL ERROR
# ==================================================

plt.figure()

plt.plot(
    time,
    error_euler,
    label="Euler"
)

plt.plot(
    time,
    error_boris,
    label="Boris"
)

plt.xlabel("Time")
plt.ylabel("Position Error")

plt.title("Position Error: Euler vs Boris")

plt.legend()
plt.grid()

plt.show()