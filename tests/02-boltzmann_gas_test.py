"""Tests for the Boltzmann gas model."""

from typing import cast
import numpy as np
import pytest

from microtherm import Gas
from microtherm.boltzmann_gas import State, Model
from microtherm.connectivity import nearest_neighbor_weights


def test_state_from_gas_levels():
    """Test conversion from hard-core gas level assignments."""
    gas = Gas(
        pos=[
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [0.74, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.99, 0.0, 0.0],
        ],
        vel=[
            [1.0, 0.0, 0.0],
            [np.sqrt(2), 0.0, 0.0],
            [2 * np.sqrt(2), 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [100.0, 0.0, 0.0],
        ],
        ids=["below", "epsilon", "four_epsilon", "middle", "high"],
    )

    state = State.from_gas_levels(gas.levels(4, 2.0), nsites=4)

    expected = np.zeros((4, 4), dtype=float)
    expected[0, 0] = 1.0 / 5.0
    expected[1, 1] = 1.0 / 5.0
    expected[2, 2] = 1.0 / 5.0
    expected[3, 2] = 1.0 / 5.0
    expected[3, 3] = 1.0 / 5.0

    assert state.levels.shape == (4, 4)
    assert state.betas.shape == (4,)
    np.testing.assert_allclose(state.levels, expected)
    np.testing.assert_allclose(state.levels.sum(), 1.0)


def test_state_from_gas_levels_rejects_invalid_assignments():
    """Test validation when converting gas level assignments."""
    with pytest.raises(ValueError, match="at least one particle"):
        State.from_gas_levels({})

    with pytest.raises(ValueError, match=r"\[site, level\] pairs"):
        State.from_gas_levels({"particle": [0]})

    with pytest.raises(ValueError, match="nonnegative"):
        State.from_gas_levels({"particle": [-1, 0]})

    with pytest.raises(ValueError, match="integers"):
        State.from_gas_levels({"particle": [0, cast(int, 1.5)]})

    with pytest.raises(ValueError, match="outside nsites=2"):
        State.from_gas_levels({"particle": [2, 0]}, nsites=2)

    with pytest.raises(ValueError, match="outside nlevels=2"):
        State.from_gas_levels({"particle": [0, 2]}, nlevels=2)


def test_state_from_gas_levels_infers_or_preserves_size():
    """Test size inference and explicit trailing empty rows or columns."""
    gas_levels = {"a": [0, 1], "b": [1, 0]}

    inferred = State.from_gas_levels(gas_levels)
    explicit = State.from_gas_levels(gas_levels, nsites=4, nlevels=3)

    assert inferred.levels.shape == (2, 2)
    assert explicit.levels.shape == (4, 3)
    np.testing.assert_allclose(inferred.levels.sum(), 1.0)
    np.testing.assert_allclose(explicit.levels.sum(), 1.0)
    np.testing.assert_allclose(explicit.levels[:2, :2], inferred.levels)
    np.testing.assert_allclose(explicit.levels[2:, :], 0.0)
    np.testing.assert_allclose(explicit.levels[:, 2:], 0.0)


def test_model_supports_arbitrary_nlevels():
    """Model supports nlevels independent from nsites."""
    levels = np.ones((4, 6), dtype=float)
    levels /= levels.sum()

    state = State(levels=levels)
    model = Model(state=state, fixed_betas={0: 1.0, 3: 0.5}, maxiter=10)

    new_state = model()

    assert new_state.levels.shape == (4, 6)
    assert cast(np.ndarray, new_state.transition).shape == (4, 6, 4, 6)
    np.testing.assert_allclose(new_state.levels.sum(), 1.0, atol=1e-12)


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
    transition = cast(np.ndarray, new_state.transition)

    assert new_state.levels.shape == (nsites, nsites)
    assert new_state.betas.shape == (nsites,)
    assert transition.shape == (nsites, nsites, nsites, nsites)

    assert np.all(np.isfinite(new_state.levels))
    assert np.all(np.isfinite(new_state.betas))
    assert np.all(np.isfinite(transition))

    assert np.all(new_state.levels >= 0.0)
    assert np.all(transition >= 0.0)

    np.testing.assert_allclose(new_state.levels.sum(), 1.0, atol=1e-12)

    # P[ell, i, m, j] must be normalized over destination states (ell, i)
    # for every source state (m, j).
    column_sums = transition.sum(axis=(0, 1))
    np.testing.assert_allclose(column_sums, 1.0, atol=1e-12)

    assert new_state.betas[0] == 1.0
    assert new_state.betas[-1] == 0.5


def test_model_accepts_precomputed_weights():
    """Test that the model can accept precomputed weights."""
    nsites = 4

    levels = np.ones((nsites, nsites), dtype=float)
    levels /= levels.sum()

    weights = nearest_neighbor_weights(nsites, nsites)

    state = State(levels=levels)

    model = Model(
        connectivity=weights,
        state=state,
        fixed_betas={0: 1.0, nsites - 1: 0.5},
        maxiter=20,
    )

    new_state = model()

    assert new_state.levels.shape == (nsites, nsites)
    assert cast(np.ndarray, new_state.transition).shape == weights.shape
    np.testing.assert_allclose(new_state.levels.sum(), 1.0, atol=1e-12)


def test_effective_diffusion_matches_site_moments():
    """Effective diffusion uses local-equilibrium site-to-site moments."""
    nsites = 4
    nlevels = 3
    dx = 1.0 / nsites
    dt = dx

    levels = np.ones((nsites, nlevels), dtype=float)
    levels /= levels.sum()

    betas = np.linspace(1.0, 2.0, nsites)
    state = State(levels=levels, betas=betas)
    model = Model(state=state, connectivity=nearest_neighbor_weights(nsites, nlevels))

    thermal, diffusive = model.effective_diffusion()

    transition = model.build_transition(betas)

    q = np.exp(-betas[:, None] * model.energies[None, :])
    q /= q.sum(axis=1, keepdims=True)

    site_transition = np.einsum("limj,mj->lm", transition, q, optimize=True)

    sites = np.arange(nsites, dtype=float)
    displacement = dx * (sites[None, :] - sites[:, None])

    expected_sigma_t = np.sum(displacement * site_transition, axis=1)
    expected_sigma_d = 0.5 * np.sum(displacement**2 * site_transition, axis=1)

    delta_beta = betas[-1] - betas[0]

    expected_thermal = nsites * expected_sigma_t / (delta_beta * dt)
    expected_diffusive = expected_sigma_d / dt

    np.testing.assert_allclose(thermal, expected_thermal)
    np.testing.assert_allclose(diffusive, expected_diffusive)

    assert thermal.shape == (nsites,)
    assert diffusive.shape == (nsites,)
    assert np.all(diffusive >= 0.0)


def test_effective_diffusion_rejects_zero_beta_drop():
    """The finite-gradient thermal coefficient is undefined at zero beta drop."""
    nsites = 3
    levels = np.ones((nsites, nsites), dtype=float)
    levels /= levels.sum()
    state = State(levels=levels, betas=np.ones(nsites))
    model = Model(state=state)

    with pytest.raises(ValueError, match="nonzero beta"):
        model.effective_diffusion()
