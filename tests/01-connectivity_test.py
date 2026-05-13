"""Tests for the module `microtherm.connectivity`."""

import numpy as np
import pytest
from microtherm.connectivity import (
    reflect_index,
    nearest_neighbor_weights,
)


def test_reflect_index():
    """Test `reflect_index()`."""
    assert reflect_index(3, 0) == 0
    assert reflect_index(3, 1) == 1
    assert reflect_index(3, 2) == 2
    assert reflect_index(3, -1) == 0
    assert reflect_index(3, -2) == 1
    assert reflect_index(3, -3) == 2
    assert reflect_index(3, 3) == 2
    assert reflect_index(3, 4) == 1
    assert reflect_index(3, 5) == 0


def test_nearest_neighbors():
    """Test `nearest_neighbor_weights()`."""
    neigh = nearest_neighbor_weights(3, 3)
    assert len(neigh) == 3
    assert neigh.ndim == 4

    assert np.all(neigh[0, :, 0, :].ravel() == 9 * [2])
    assert np.all(neigh[1, :, 0, :].ravel() == 9 * [1])
    assert np.all(neigh[2, :, 0, :].ravel() == 9 * [0])

    assert np.all(neigh[2, :, 2, :].ravel() == 9 * [2])
    assert np.all(neigh[1, :, 2, :].ravel() == 9 * [1])
    assert np.all(neigh[0, :, 2, :].ravel() == 9 * [0])

    assert np.all(neigh[:, :, 1, :].ravel() == 27 * [1])


def test_nearest_neighbors_rejects_nonpositive_sizes():
    """Test `nearest_neighbor_weights()` argument validation."""
    with pytest.raises(ValueError, match="nsites must be positive"):
        nearest_neighbor_weights(0, 1)
    with pytest.raises(ValueError, match="nlevels must be positive"):
        nearest_neighbor_weights(1, 0)


def test_transition_tensor_uses_source_energy_in_nonequilibrium():
    """The source-energy factor must not cancel for site-dependent beta."""
    from microtherm.connectivity import transition_tensor

    weights = np.ones((2, 2, 2, 2), dtype=float)
    beta = np.array([0.7, 1.3])
    energy = np.array([0.0, 1.0])

    transition = transition_tensor(weights, beta, energy)

    expected = weights * np.exp(
        beta[:, None, None, None]
        * (energy[None, None, None, :] - energy[None, :, None, None])
    )
    expected /= expected.sum(axis=(0, 1), keepdims=True)

    np.testing.assert_allclose(transition, expected)
    np.testing.assert_allclose(transition.sum(axis=(0, 1)), 1.0)


def test_transition_weights_inverts_isothermal_transition_tensor():
    """Equilibrium transition probabilities recover normalized weights."""
    from microtherm.connectivity import transition_tensor, transition_weights

    raw = nearest_neighbor_weights(3, 4)
    weights = raw / raw.sum(axis=(0, 1), keepdims=True)
    energy = np.arange(4, dtype=float) ** 2 / 2.0
    beta = 0.8

    transition = transition_tensor(weights, np.full(3, beta), energy)
    recovered = transition_weights(transition, beta, energy)

    np.testing.assert_allclose(recovered, weights, rtol=1e-12, atol=1e-12)
