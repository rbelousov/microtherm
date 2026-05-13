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
