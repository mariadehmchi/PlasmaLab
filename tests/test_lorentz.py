import numpy as np

from plasmalab.particles.particle import Particle
from plasmalab.fields.uniform import (
    UniformElectricField,
    UniformMagneticField
)
from plasmalab.physics.lorentz import lorentz_force


def test_lorentz_force_with_zero_fields():
    """
    If E = 0 and B = 0,
    the Lorentz force must be zero.
    """

    particle = Particle(
        mass=1.0,
        charge=1.0,
        position=[0.0, 0.0, 0.0],
        velocity=[1.0, 2.0, 3.0]
    )

    electric_field = UniformElectricField(
        [0.0, 0.0, 0.0]
    )

    magnetic_field = UniformMagneticField(
        [0.0, 0.0, 0.0]
    )

    force = lorentz_force(
        particle,
        electric_field,
        magnetic_field,
        time=0.0
    )

    expected_force = np.array(
        [0.0, 0.0, 0.0]
    )

    assert np.allclose(
        force,
        expected_force
    )

def test_lorentz_force_with_electric_field():

    particle = Particle(
        mass=1.0,
        charge=2.0,
        position=[0.0, 0.0, 0.0],
        velocity=[0.0, 0.0, 0.0]
    )

    electric_field = UniformElectricField(
        [1.0, 2.0, 3.0]
    )

    magnetic_field = UniformMagneticField(
        [0.0, 0.0, 0.0]
    )

    force = lorentz_force(
        particle,
        electric_field,
        magnetic_field,
        time=0.0
    )

    expected_force = np.array(
        [2.0, 4.0, 6.0]
    )

    assert np.allclose(
        force,
        expected_force
    )
def test_lorentz_force_with_magnetic_field():

    particle = Particle(
        mass=1.0,
        charge=1.0,
        position=[0.0, 0.0, 0.0],
        velocity=[1.0, 0.0, 0.0]
    )

    electric_field = UniformElectricField(
        [0.0, 0.0, 0.0]
    )

    magnetic_field = UniformMagneticField(
        [0.0, 0.0, 1.0]
    )

    force = lorentz_force(
        particle,
        electric_field,
        magnetic_field,
        time=0.0
    )

    expected_force = np.array(
        [0.0, -1.0, 0.0]
    )

    assert np.allclose(
        force,
        expected_force
    )
def test_magnetic_force_is_perpendicular_to_velocity():

    particle = Particle(
        mass=1.0,
        charge=1.0,
        position=[0.0, 0.0, 0.0],
        velocity=[1.0, 2.0, 3.0]
    )

    electric_field = UniformElectricField(
        [0.0, 0.0, 0.0]
    )

    magnetic_field = UniformMagneticField(
        [2.0, -1.0, 4.0]
    )

    force = lorentz_force(
        particle,
        electric_field,
        magnetic_field,
        time=0.0
    )

    dot_product = np.dot(
        force,
        particle.velocity
    )

    assert np.isclose(
        dot_product,
        0.0
    )