"""Connectivity weights for a Boltzmann-gas model.

The dynamical graph is represented by weights

    w_IJ = g_I * a_IJ * gamma_IJ

where I = (target_site, target_level) and J = (source_site, source_level).
The Markov tensor is normalized over destination states I for every source
state J.
"""

import numpy as np


def reflect_index(nsites: int, target: int) -> int:
    """Reflect a target site at half-integer walls.

    Walls are located at -0.5 and nsites - 0.5, so

        -1 -> 0
        -2 -> 1
         nsites -> nsites - 1
         nsites + 1 -> nsites - 2

    Args:
        nsites: Number of lattice sites.
        target: Possibly out-of-bounds target site.

    Returns:
        Reflected in-bounds site index.
    """
    if nsites <= 1:
        return 0

    period = 2 * nsites
    r = target % period

    if r < nsites:
        return int(r)

    return int(period - 1 - r)


def nearest_neighbor_weights(nsites: int, nlevels: int, reflect = True) -> np.ndarray:
    """Nearest-neighbor transition weights with optional reflective path multiplicities.

    The nearest-neighbor graph allows transitions between sites with |ell - m| <= 1
    before reflection. Reflection applied by default does not merely clip the target site:
    it increases the path multiplicity gamma_IJ when multiple attempted moves
    land on the same reflected target.

    All destination and source energy-level pairs are structurally allowed.
    The energy dependence enters later through the microcaliber factor
    exp(-beta_ell * (epsilon_i - epsilon_j)), not here.

    Args:
        nsites: Number of lattice sites.
        nlevels: Number of energy levels.
        reflect: If True (default), reflective boundary conditions apply.

    Returns:
        Weight tensor w[ell, i, m, j].
    """
    if nsites <= 0:
        raise ValueError(f"nsites must be positive, got {nsites}")
    if nlevels <= 0:
        raise ValueError(f"nlevels must be positive, got {nlevels}")

    weights = np.zeros((nsites, nlevels, nsites, nlevels), dtype=float)

    for source_site in range(nsites):
        if reflect:
            # Attempted nearest-neighbor displacements. Reflections may map
            # multiple attempts to the same target_site, giving gamma > 1.
            target_sites = [
                reflect_index(nsites, source_site + delta) for delta in (-1, 0, 1)
            ]
        else:
            target_sites = []
            for delta in (-1, 0, 1):
                target_site = source_site + delta
                if 0 <= target_site and target_site < nsites:
                    target_sites.append(target_site)

        for target_site in target_sites:
            # All target/source energy-level pairs have the same structural
            # graph weight. The target energy preference is applied later
            # in the Markov kernel.
            weights[target_site, :, source_site, :] += 1.0

    return weights


def flat_matrix(tensor: np.ndarray) -> np.ndarray:
    """Flatten 4D tensor[ell, i, m, j] into a matrix[I, J].

    The flattened state index is

        I = ell * nlevels + i.

    Args:
        tensor: Four-dimensional tensor

    Returns:
        Two-dimensional matrix
    """
    nsites, nlevels, _, _ = tensor.shape
    return tensor.reshape(nsites * nlevels, nsites * nlevels)


def _validate_transition_inputs(
    weights: np.ndarray, beta: np.ndarray, energy: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return validated transition inputs as floating-point arrays."""
    weights = np.asarray(weights, dtype=float)
    beta = np.asarray(beta, dtype=float)
    energy = np.asarray(energy, dtype=float)

    if weights.ndim != 4:
        raise ValueError(
            "weights must have shape (nsites, nlevels, nsites, nlevels); "
            f"got {weights.shape}"
        )

    nsites, nlevels, nsites_2, nlevels_2 = weights.shape
    if nsites != nsites_2 or nlevels != nlevels_2:
        raise ValueError(
            "weights must have shape (nsites, nlevels, nsites, nlevels); "
            f"got {weights.shape}"
        )

    if beta.shape != (nsites,):
        raise ValueError(f"beta must have shape ({nsites},), got {beta.shape}")

    if energy.shape != (nlevels,):
        raise ValueError(f"energy must have shape ({nlevels},), got {energy.shape}")

    return weights, beta, energy


def transition_tensor(
    weights: np.ndarray,
    beta: np.ndarray,
    energy: np.ndarray,
    degeneracy: np.ndarray | None = None,
) -> np.ndarray:
    """Build the Markov transition tensor P[ell, i, m, j].

    For each source state J = (m, j), this computes the microcaliber
    transition probability

        P_{ell,i|m,j} =
            w_{ell,i|m,j} exp[-beta_ell (epsilon_i - epsilon_j)] / zeta_{m,j}

    where zeta_{m,j} normalizes over destination states (ell, i).

    The source-energy term cannot be dropped when beta varies with the
    destination site ell. It cancels only in the isothermal case.

    Args:
        weights: Structural weights w[ell, i, m, j].
        beta: Inverse temperatures beta[ell].
        energy: Energy levels epsilon[i].
        degeneracy: Optional destination degeneracies g[i] or g[ell, i].
            If omitted, degeneracies are assumed to be included in weights or one.

    Returns:
        Markov transition tensor P[ell, i, m, j], normalized over (ell, i)
        for every source state (m, j).
    """
    weights, beta, energy = _validate_transition_inputs(weights, beta, energy)
    nsites, nlevels, _, _ = weights.shape

    # factor[ell, i, j] = exp[-beta_ell * (epsilon_i - epsilon_j)]
    #                      = exp[ beta_ell * (epsilon_j - epsilon_i)]
    exponent = beta[:, None, None] * (energy[None, None, :] - energy[None, :, None])
    factor = np.exp(exponent)

    if degeneracy is not None:
        degeneracy = np.asarray(degeneracy, dtype=float)

        if degeneracy.shape == (nlevels,):
            factor = factor * degeneracy[None, :, None]
        elif degeneracy.shape == (nsites, nlevels):
            factor = factor * degeneracy[:, :, None]
        else:
            raise ValueError(
                "degeneracy must have shape "
                f"({nlevels},) or ({nsites}, {nlevels}), got {degeneracy.shape}"
            )

    unnormalized = weights * factor[:, :, None, :]
    zeta = unnormalized.sum(axis=(0, 1), keepdims=True)

    return np.divide(
        unnormalized,
        zeta,
        out=np.zeros_like(unnormalized),
        where=zeta > 0,
    )


def transition_weights(
    transition: np.ndarray,
    beta: float,
    energy: np.ndarray,
    *,
    atol: float = 1e-12,
) -> np.ndarray:
    """Infer normalized weights from equilibrium transition probabilities.

    This solves the equilibrium version of Eq. (S1),

        P_{ell,i|m,j} ∝ w_{ell,i|m,j} exp[-beta * (epsilon_i - epsilon_j)],

    with the normalization sum_{ell,i} w[ell, i, m, j] = 1. Because beta is
    constant in the equilibrium calibration run, exp(beta * epsilon_j) cancels
    for each fixed source state (m, j), leaving

        w_{ell,i|m,j} ∝ P_{ell,i|m,j} exp(beta * epsilon_i).

    Args:
        transition: Measured transition probabilities P[ell, i, m, j].
        beta: Equilibrium inverse temperature used to measure transition.
        energy: Energy levels epsilon[i].
        atol: Absolute tolerance for identifying zero-probability source states.

    Returns:
        Normalized weight tensor w[ell, i, m, j].
    """
    transition = np.asarray(transition, dtype=float)
    energy = np.asarray(energy, dtype=float)

    if transition.ndim != 4:
        raise ValueError(
            "transition must have shape (nsites, nlevels, nsites, nlevels); "
            f"got {transition.shape}"
        )

    nsites, nlevels, nsites_2, nlevels_2 = transition.shape
    if nsites != nsites_2 or nlevels != nlevels_2:
        raise ValueError(
            "transition must have shape (nsites, nlevels, nsites, nlevels); "
            f"got {transition.shape}"
        )

    if energy.shape != (nlevels,):
        raise ValueError(f"energy must have shape ({nlevels},), got {energy.shape}")

    if np.any(transition < -atol):
        raise ValueError("transition contains negative probabilities")

    transition = np.clip(transition, 0.0, None)
    raw = transition * np.exp(float(beta) * energy)[None, :, None, None]
    norm = raw.sum(axis=(0, 1), keepdims=True)

    return np.divide(raw, norm, out=np.zeros_like(raw), where=norm > atol)
