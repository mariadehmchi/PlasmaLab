import numpy as np


class Simulation:
    """
    Controls the time evolution of a charged particle.

    Parameters
    ----------
    particle : Particle
        Particle to simulate.

    electric_field : object
        Electric field.

    magnetic_field : object
        Magnetic field.

    integrator : function
        Numerical integrator used to advance the particle.

    dt : float
        Simulation time step.

    steps : int
        Number of simulation steps.
    """

    def __init__(
        self,
        particle,
        electric_field,
        magnetic_field,
        integrator,
        dt,
        steps
    ):

        self.particle = particle
        self.electric_field = electric_field
        self.magnetic_field = magnetic_field
        self.integrator = integrator

        self.dt = float(dt)
        self.steps = int(steps)

        self.time = 0.0

        self.times = []
        self.positions = []
        self.velocities = []
        self.energies = []

    def kinetic_energy(self):
        """
        Calculate the kinetic energy
        of the current particle.
        """

        return (
            0.5
            * self.particle.mass
            * np.dot(
                self.particle.velocity,
                self.particle.velocity
            )
        )

    def store_state(self):
        """
        Store the current state of the particle.
        """

        self.times.append(self.time)

        self.positions.append(
            self.particle.position.copy()
        )

        self.velocities.append(
            self.particle.velocity.copy()
        )

        self.energies.append(
            self.kinetic_energy()
        )

    def run(self):
        """
        Run the simulation.
        """

        # Store initial state at t = 0
        self.store_state()

        for _ in range(self.steps):

            # Advance particle
            self.integrator(
                self.particle,
                self.electric_field,
                self.magnetic_field,
                self.time,
                self.dt
            )

            # Update time
            self.time += self.dt

            # Store new state
            self.store_state()

        return (
            np.array(self.times),
            np.array(self.positions),
            np.array(self.velocities),
            np.array(self.energies)
        )
