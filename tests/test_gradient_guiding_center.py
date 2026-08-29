import numpy as np

from plasmalab.particles.particle import Particle
from plasmalab.fields.gradient import GradientMagneticField
from plasmalab.diagnostics.gradient_guiding_center import (
    guiding_center_position
)


def test_guiding_center_position():

    particle = Particle(
        mass=1.0,
        charge=1.0,
        position=[0.0, 0.0, 0.0],
        velocity=[1.0, 0.0, 0.0]
    )

    magnetic_field = GradientMagneticField(
        B0=1.0,
        alpha=0.05
    )

    R_gc = guiding_center_position(
        particle,
        magnetic_field
    )

    # At x = 0:
    # B = (0, 0, 1)
    # Omega = qB/m = 1
    #
    # v x B = (1,0,0) x (0,0,1)
    #       = (0,-1,0)
    #
    # Therefore:
    # R_gc = (0,-1,0)

    expected = np.array([
        0.0,
        -1.0,
        0.0
    ])

    assert np.allclose(
        R_gc,
        expected
    )