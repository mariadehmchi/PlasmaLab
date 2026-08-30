import numpy as np

from plasmalab.particles.particle import Particle
from plasmalab.fields.mirror import MirrorMagneticField
from plasmalab.diagnostics.magnetic_mirror import magnetic_moment


def test_magnetic_moment_for_perpendicular_motion():
    field = MirrorMagneticField(
        B0=1.0,
        alpha=0.1
    )

    particle = Particle(
        mass=1.0,
        charge=1.0,
        position=[0.0, 0.0, 0.0],
        velocity=[1.0, 0.0, 0.0]
    )

    mu = magnetic_moment(
        particle,
        field,
        particle.position,
        particle.velocity,
        0.0
    )

    assert np.isclose(
        mu,
        0.5
    )


def test_magnetic_moment_for_parallel_motion():
    field = MirrorMagneticField(
        B0=1.0,
        alpha=0.1
    )

    particle = Particle(
        mass=1.0,
        charge=1.0,
        position=[0.0, 0.0, 0.0],
        velocity=[0.0, 0.0, 1.0]
    )

    mu = magnetic_moment(
        particle,
        field,
        particle.position,
        particle.velocity,
        0.0
    )

    assert np.isclose(
        mu,
        0.0
    )