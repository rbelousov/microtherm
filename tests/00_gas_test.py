"""Unit tests for `microtherm.gas` module."""

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
