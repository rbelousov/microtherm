"""Hard-core gas data.

Attributes:
    DIM (int): Number of physical dimensions
"""

import bz2
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import cast, Self
from collections.abc import Iterable

DIM: int = 3


class State:
    """State of an ideal gas.

    Attributes:
        box (numpy.typing.NDArray): Confining box
        ids (list[str]): Particles' IDs
        pos (numpy.typing.NDArray): Collection of particles' positions
        vel (numpy.typing.NDArray): Collection of particles' velocities
    """

    def __init__(
        self,
        pos: ArrayLike,
        vel: ArrayLike,
        box: ArrayLike = [1, 1, 1],
        ids: list[str] | None = None,
    ):
        """Initialize state.

        Args:
            pos: Particles' positions
            vel: Particles' velocities
            box: Confining box
            ids: Particles' IDs

        Raises:
            AssertionError: on inconsistent lenghts of positions and velocities
        """
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)
        self.box = np.array(box, dtype=float)

        sz = len(self.pos)
        self.ids = [str(id) for id in cast(Iterable, range(sz) if ids is None else ids)]

        assert sz == len(self.vel), (
            f"Inconsistent sizes of positions ({sz}) and velocities ({len(self.vel)})"
        )
        assert sz == len(self.ids), (
            f"Inconsistent sizes of positions ({sz}) and IDs ({len(self.ids)})"
        )

    def save(self, path: str):
        """Save data to a file.

        Args:
            path: Path to the file
        """
        np.savez(path, pos=self.pos, vel=self.vel, box=self.box, ids=np.array(self.ids))

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load data from a file.

        Args:
            path: Path to the file

        Returns:
            Loaded state
        """
        data = np.load(path)
        return cls(
            ids=data["ids"].tolist(), box=data["box"], pos=data["pos"], vel=data["vel"]
        )

    @classmethod
    def from_dynamo_config(cls, path: str | Path) -> Self:
        """Load data from an XML or archived XML configuration file for dynamO.

        Args:
            path: Path to the file

        Returns:
            Loaded state
        """
        if str(path).endswith(".bz2"):
            with bz2.open(path, "rb") as f:
                tree = ET.parse(f)
        else:
            tree = ET.parse(path)
        root = tree.getroot()

        sim_size: ET.Element = cast(ET.Element, root.find(".//SimulationSize"))
        domain = np.array(
            [
                float(cast(str, sim_size.get("x"))),
                float(cast(str, sim_size.get("y"))),
                float(cast(str, sim_size.get("z"))),
            ],
            dtype=float,
        )
        diameters = [
            float(cast(str, wall.get("Diameter")))
            for wall in root.findall('.//Local[@Type="Wall"]')
        ]
        diameter = np.mean(diameters)
        box = domain.copy()
        box[0] -= np.sum(diameters)

        # All particles live under ParticleData/Pt with <P> (position) and <V> (velocity)
        pts = root.findall(".//Pt")
        sz = len(pts)
        ids = []
        pos = np.empty((sz, 3), dtype=float)
        vel = np.empty((sz, 3), dtype=float)

        for idx, pt in enumerate(pts):
            ids.append(cast(str, pt.get("ID")))
            p = cast(ET.Element, pt.find("P"))
            v = cast(ET.Element, pt.find("V"))
            pos[idx, :] = [
                float(cast(str, p.get("x"))),
                float(cast(str, p.get("y"))),
                float(cast(str, p.get("z"))),
            ]
            vel[idx, :] = [
                float(cast(str, v.get("x"))),
                float(cast(str, v.get("y"))),
                float(cast(str, v.get("z"))),
            ]
        pos += box / 2
        pos[:, 1:] %= box[1:]
        eps = 1e-2 * diameter
        assert np.all(pos < box + eps) and np.all(pos > -eps), (
            f"Gas state does not fit the simulation box {box}"
        )

        return cls(ids=ids, box=box, pos=pos, vel=vel)

    def __len__(self) -> int:
        """Return number of particles."""
        return len(self.pos)

    def levels(
        self, sz: int, epsilon: float, mx: int | None = None
    ) -> dict[str, list[int]]:
        """Return particles' spatial bins and energy levels.

        Args:
            sz: Number of spatial bins and energy levels
            epsilon: Energy coefficient
            mx: Optional maximum energy level

        Returns:
            Mapping from particle ID to ``[bin, level]``

        Raises:
            ValueError: On non-positive number of bins or energy coefficient
        """
        if sz <= 0:
            raise ValueError("Number of bins and levels must be positive")
        if epsilon <= 0:
            raise ValueError("Energy coefficient must be positive")
        mx = sz if mx is None else mx
        if mx <= 0:
            raise ValueError("Maximum energy level must be positive")

        bin_width = self.box[0] / sz
        bins = np.floor(self.pos[:, 0] / bin_width).astype(int)
        bins = np.clip(bins, 0, sz - 1)

        energies = np.sum(self.vel**2, axis=1) / 2
        upper_bounds = epsilon * np.arange(1, mx) ** 2 / 2
        levels = np.searchsorted(upper_bounds, energies, side="right")

        return {
            id: [int(bin), int(level)]
            for id, bin, level in zip(self.ids, bins, levels, strict=True)
        }

    def site_occupancies(self, nbins: int) -> NDArray[np.floating]:
        """Return particles' distributions along the x axis subdivided into bins.

        Args:
            nbins: Number of bins to use

        Returns:
            Histogram probabilities
        """
        cs, _ = np.histogram(self.pos[:, 0], bins=nbins, range=(0, self.box[0]))
        return cs / len(self)

    def site_energies(self, nbins: int) -> NDArray[np.floating]:
        """Return particles'  energy distributions along the x axis subdivided into bins.

        Args:
            nbins: Number of bins to use

        Returns:
            Histogram probabilities  weighted by energy
        """
        ws = np.sum(self.vel**2, axis=1) / 2
        es, _ = np.histogram(
            self.pos[:, 0], bins=nbins, range=(0, self.box[0]), weights=ws
        )

        return es / len(self)

    def energy_profile(self, nbins: int) -> NDArray[np.floating]:
        """Return system's energy profile."""
        energies = self.site_energies(nbins)
        return energies / np.sum(energies)

    def kinetic_temperature(self) -> np.floating:
        """Return the kinetic temperature of the whole system."""
        return np.mean(self.vel**2)

    def kinetic_temperatures(self, nbins: int) -> NDArray[np.floating]:
        """Return temperature profile along the x axis subdivided into bins.

        Args:
            nbins: Number of bins to use

        Returns:
            Histogram probabilities  weighted by energy
        """
        ws = np.sum(self.vel**2, axis=1) / 2
        es, _ = np.histogram(
            self.pos[:, 0], bins=nbins, range=(0, self.box[0]), weights=ws
        )
        cs, _ = np.histogram(self.pos[:, 0], bins=nbins, range=(0, self.box[0]))
        # Avoid division by zero
        avg_es = np.divide(es, cs, out=np.zeros_like(es), where=cs != 0)

        return 2.0 * avg_es / DIM
