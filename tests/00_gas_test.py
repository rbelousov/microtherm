"""Unit tests for `microtherm.gas` module."""

import numpy as np
import pytest
from microtherm import Gas
from pathlib import Path


@pytest.fixture
def dynamo_config() -> Path:
    """Return path to a fixture DynamO configuration file."""
    return Path(__file__).parent / "data" / "dynamo_config.xml"


def test_state(dynamo_config):
    """Test `microtherm.gas.State`."""
    state = Gas.from_dynamo_config(dynamo_config)
    assert len(state) == 8
    assert state.box == pytest.approx([1.0, 0.4, 0.4])


def test_state_levels():
    """Test spatial bins and energy levels."""
    state = Gas(
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

    assert state.levels(4, 2.0) == {
        "below": [0, 0],
        "epsilon": [1, 1],
        "four_epsilon": [2, 2],
        "middle": [3, 2],
        "high": [3, 3],
    }


def test_state_levels_rejects_non_positive_parameters():
    """Test that levels validates its parameters."""
    state = Gas(pos=[[0.0, 0.0, 0.0]], vel=[[0.0, 0.0, 0.0]])

    with pytest.raises(ValueError):
        state.levels(0, 1.0)

    with pytest.raises(ValueError):
        state.levels(1, 0.0)
