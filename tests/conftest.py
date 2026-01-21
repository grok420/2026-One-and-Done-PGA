"""
Shared pytest fixtures for PGA One and Done tests.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_db_path(temp_dir):
    """Create a temporary database path."""
    return temp_dir / "test_data.db"


@pytest.fixture
def mock_env_no_api_key():
    """Mock environment with no API key."""
    with patch.dict(os.environ, {
        "DATAGOLF_API_KEY": "",
        "PGA_OAD_EMAIL": "",
        "PGA_OAD_PASSWORD": "",
        "PGA_OAD_USERNAME": "",
    }, clear=False):
        yield


@pytest.fixture
def mock_env_with_api_key():
    """Mock environment with API key set."""
    with patch.dict(os.environ, {
        "DATAGOLF_API_KEY": "test_api_key_12345",
        "PGA_OAD_EMAIL": "test@example.com",
        "PGA_OAD_PASSWORD": "testpass",
        "PGA_OAD_USERNAME": "testuser",
    }, clear=False):
        yield


@pytest.fixture
def mock_requests_session():
    """Mock requests.Session for API tests."""
    with patch("requests.Session") as mock_session:
        yield mock_session


@pytest.fixture
def sample_golfer_data():
    """Sample golfer data for testing."""
    return {
        "name": "Scottie Scheffler",
        "owgr": 1,
        "datagolf_id": "12345",
        "stats": {
            "sg_total": 2.5,
            "sg_off_tee": 0.8,
            "sg_approach": 1.0,
            "sg_around_green": 0.4,
            "sg_putting": 0.3,
            "driving_distance": 305.5,
            "driving_accuracy": 62.5,
        }
    }


@pytest.fixture
def sample_prediction_response():
    """Sample API prediction response."""
    return {
        "event_name": "The Masters",
        "baseline_history_fit": [
            {
                "player_name": "Scottie Scheffler",
                "dg_id": "12345",
                "win_prob": 0.15,
                "top_5_prob": 0.35,
                "top_10_prob": 0.50,
                "top_20_prob": 0.70,
                "make_cut_prob": 0.95,
                "expected_place": 8.5,
            },
            {
                "player_name": "Rory McIlroy",
                "dg_id": "12346",
                "win_prob": 0.10,
                "top_5_prob": 0.30,
                "top_10_prob": 0.45,
                "top_20_prob": 0.65,
                "make_cut_prob": 0.92,
                "expected_place": 12.0,
            },
        ]
    }
