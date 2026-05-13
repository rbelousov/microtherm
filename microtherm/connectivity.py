"""Connectivity weights for a Boltzmann-gas model.

The dynamical graph is represented by weights

    w_IJ = g_I * a_IJ * gamma_IJ

where I = (target_site, target_level) and J = (source_site, source_level).
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


def nearest_neighbor_weights(nsites: int, nlevels: int) -> np.ndarray:
    """Nearest-neighbor transition weights with reflective path multiplicities.

    The nearest-neighbor graph allows transitions between sites with |ell - m| <= 1
    before reflection. Reflection does not merely clip the target site: it
    increases the path multiplicity gamma_IJ when multiple attempted moves
    land on the same reflected target.

    All destination energy levels are allowed. The energy dependence enters
    later through the Boltzmann factor exp(-beta_ell * epsilon_i), not here.

    Args:
        nsites: Number of lattice sites.
        nlevels: Number of energy levels.

    Returns:
        Weight tensor w[ell, i, m, j].
    """
    if nsites <= 0:
        raise ValueError(f"nsites must be positive, got {nsites}")
    if nlevels <= 0:
        raise ValueError(f"nlevels must be positive, got {nlevels}")

    weights = np.zeros((nsites, nlevels, nsites, nlevels), dtype=float)

    for source_site in range(nsites):
        # Attempted nearest-neighbor displacements. Reflections may map
        # multiple attempts to the same target_site, giving gamma > 1.
        target_sites = [
            reflect_index(nsites, source_site + delta) for delta in (-1, 0, 1)
        ]

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


def transition_tensor(
    weights: np.ndarray,
    beta: np.ndarray,
    energy: np.ndarray,
    degeneracy: np.ndarray | None = None,
) -> np.ndarray:
    """Build the Markov transition matrix P_IJ from structural weights.

    For each source state J, this computes

        P_IJ = w_IJ exp(-beta_ell epsilon_i) / zeta_J

    where

        zeta_J = sum_I w_IJ exp(-beta_ell epsilon_i).

    Args:
        weights: Structural weights w[ell, i, m, j].
        beta: Inverse temperatures beta[ell].
        energy: Energy levels epsilon[i].
        degeneracy: Optional destination degeneracies g[ell, i] or g[i].
            If omitted, all degeneracies are one.

    Returns:
        Markov transition tensor P[ell, i, m, j].
    """
    nsites, nlevels, _, _ = weights.shape

    beta = np.asarray(beta, dtype=float)
    energy = np.asarray(energy, dtype=float)

    if beta.shape != (nsites,):
        raise ValueError(f"beta must have shape ({nsites},), got {beta.shape}")

    if energy.shape != (nlevels,):
        raise ValueError(f"energy must have shape ({nlevels},), got {energy.shape}")

    boltzmann = np.exp(-beta[:, None] * energy[None, :])

    if degeneracy is not None:
        degeneracy = np.asarray(degeneracy, dtype=float)

        if degeneracy.shape == (nlevels,):
            boltzmann = boltzmann * degeneracy[None, :]
        elif degeneracy.shape == (nsites, nlevels):
            boltzmann = boltzmann * degeneracy
        else:
            raise ValueError(
                "degeneracy must have shape "
                f"({nlevels},) or ({nsites}, {nlevels}), got {degeneracy.shape}"
            )

    unnormalized = weights * boltzmann[:, :, None, None]
    zeta = unnormalized.sum(axis=(0, 1), keepdims=True)

    # with np.errstate(divide="ignore", invalid="ignore"):
    probs = np.divide(
        unnormalized,
        zeta,
        out=np.zeros_like(unnormalized),
        where=zeta > 0,
    )

    return probs
