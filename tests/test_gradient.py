import numpy as np

from plasmalab.fields.gradient import GradientMagneticField


def test_gradient_field_at_origin():
    field = GradientMagneticField(1.0, 0.05)

    B = field.value(np.array([0.0, 0.0, 0.0]), 0.0)

    np.testing.assert_allclose(
        B,
        np.array([0.0, 0.0, 1.0])
    )


def test_gradient_field_increases_with_x():
    field = GradientMagneticField(1.0, 0.05)

    B = field.value(np.array([1.0, 0.0, 0.0]), 0.0)

    np.testing.assert_allclose(
        B,
        np.array([0.0, 0.0, 1.05])
    )


def test_gradient_field_decreases_with_negative_x():
    field = GradientMagneticField(1.0, 0.05)

    B = field.value(np.array([-1.0, 0.0, 0.0]), 0.0)

    np.testing.assert_allclose(
        B,
        np.array([0.0, 0.0, 0.95])
    )