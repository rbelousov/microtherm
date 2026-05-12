"""Tests for the Boltzmann gas model."""

import numpy as np

from microtherm.boltzmann_gas import State, Model
from microtherm.connectivity import nearest_neighbor_weights


def test_model_preserves_probability_and_shapes():
    """Minimum smoke test for the Boltzmann gas model."""
    nsites = 5

    levels = np.ones((nsites, nsites), dtype=float)
    levels /= levels.sum()

    state = State(nsites=nsites, levels=levels)

    model = Model(
        state=state,
        connectivity=None,
        fixed_betas={0: 1.0, nsites - 1: 0.5},
        tol=1e-9,
        beta_tol=1e-9,
        maxiter=20,
    )

    new_state = model()

    assert new_state.levels.shape == (nsites, nsites)
    assert new_state.betas.shape == (nsites,)
    assert new_state.transition.shape == (nsites, nsites, nsites, nsites)

    assert np.all(np.isfinite(new_state.levels))
    assert np.all(np.isfinite(new_state.betas))
    assert np.all(np.isfinite(new_state.transition))

    assert np.all(new_state.levels >= 0.0)
    assert np.all(new_state.transition >= 0.0)

    np.testing.assert_allclose(new_state.levels.sum(), 1.0, atol=1e-12)

    # P[ell, i, m, j] must be normalized over destination states (ell, i)
    # for every source state (m, j).
    column_sums = new_state.transition.sum(axis=(0, 1))
    np.testing.assert_allclose(column_sums, 1.0, atol=1e-12)

    assert new_state.betas[0] == 1.0
    assert new_state.betas[-1] == 0.5


def test_model_accepts_precomputed_weights():
    """Test that the model can accept precomputed weights."""
    nsites = 4

    levels = np.ones((nsites, nsites), dtype=float)
    levels /= levels.sum()

    weights = nearest_neighbor_weights(nsites)

    state = State(levels=levels)

    model = Model(
        connectivity=weights,
        state=state,
        fixed_betas={0: 1.0, nsites - 1: 0.5},
        maxiter=20,
    )

    new_state = model()

    assert new_state.levels.shape == (nsites, nsites)
    assert new_state.transition.shape == weights.shape
    np.testing.assert_allclose(new_state.levels.sum(), 1.0, atol=1e-12)
