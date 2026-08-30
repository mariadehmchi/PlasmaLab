import numpy as np

from plasmalab.fields.mirror import MirrorMagneticField


def test_mirror_field_at_origin():
    field = MirrorMagneticField(
        B0=1.0,
        alpha=0.1
    )

    B = field.value(
        np.array([0.0, 0.0, 0.0]),
        0.0
    )

    assert np.allclose(
        B,
        np.array([0.0, 0.0, 1.0])
    )


def test_mirror_field_is_symmetric():
    field = MirrorMagneticField(
        B0=1.0,
        alpha=0.1
    )

    B_plus = field.value(
        np.array([0.0, 0.0, 2.0]),
        0.0
    )

    B_minus = field.value(
        np.array([0.0, 0.0, -2.0]),
        0.0
    )

    assert np.allclose(B_plus, B_minus)


def test_mirror_field_increases_away_from_center():
    field = MirrorMagneticField(
        B0=1.0,
        alpha=0.1
    )

    B0 = field.value(
        np.array([0.0, 0.0, 0.0]),
        0.0
    )

    B1 = field.value(
        np.array([0.0, 0.0, 1.0]),
        0.0
    )

    B2 = field.value(
        np.array([0.0, 0.0, 2.0]),
        0.0
    )

    assert B1[2] > B0[2]
    assert B2[2] > B1[2]


def test_mirror_field_has_transverse_components_off_axis():
    field = MirrorMagneticField(
        B0=1.0,
        alpha=0.1
    )

    B = field.value(
        np.array([1.0, 0.0, 1.0]),
        0.0
    )

    assert np.isclose(B[0], -0.1)
    assert np.isclose(B[1], 0.0)
    assert np.isclose(B[2], 1.1)


def test_mirror_field_is_divergence_free():
    field = MirrorMagneticField(
        B0=1.0,
        alpha=0.1
    )

    x = 1.0
    y = 2.0
    z = 1.5

    h = 1e-5

    position = np.array([x, y, z])

    B_x_plus = field.value(
        position + np.array([h, 0.0, 0.0]),
        0.0
    )

    B_x_minus = field.value(
        position - np.array([h, 0.0, 0.0]),
        0.0
    )

    B_y_plus = field.value(
        position + np.array([0.0, h, 0.0]),
        0.0
    )

    B_y_minus = field.value(
        position - np.array([0.0, h, 0.0]),
        0.0
    )

    B_z_plus = field.value(
        position + np.array([0.0, 0.0, h]),
        0.0
    )

    B_z_minus = field.value(
        position - np.array([0.0, 0.0, h]),
        0.0
    )

    dBx_dx = (
        B_x_plus[0] - B_x_minus[0]
    ) / (2.0 * h)

    dBy_dy = (
        B_y_plus[1] - B_y_minus[1]
    ) / (2.0 * h)

    dBz_dz = (
        B_z_plus[2] - B_z_minus[2]
    ) / (2.0 * h)

    divergence = dBx_dx + dBy_dy + dBz_dz

    assert np.isclose(
        divergence,
        0.0,
        atol=1e-8
    )