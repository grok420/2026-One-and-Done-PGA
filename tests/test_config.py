"""
Tests for config.py - Configuration management.
"""

import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config, get_config, get_schedule, get_tournament_by_name


class TestConfig:
    """Tests for Config class."""

    def test_config_loads_defaults(self, temp_dir):
        """Test that config loads with default values."""
        with patch.dict(os.environ, {
            "DATAGOLF_API_KEY": "",
            "PGA_OAD_EMAIL": "",
            "PGA_OAD_PASSWORD": "",
            "PGA_OAD_USERNAME": "",
        }, clear=False):
            with patch.object(Config, '__post_init__', lambda self: None):
                config = Config()
                assert config.league_name == "Bushwood"
                assert config.risk_level == 5
                assert config.default_simulations == 50000

    def test_config_loads_env_vars(self, temp_dir):
        """Test that config loads environment variables."""
        with patch.dict(os.environ, {
            "DATAGOLF_API_KEY": "test_key",
            "PGA_OAD_EMAIL": "test@test.com",
            "PGA_OAD_PASSWORD": "testpass",
            "PGA_OAD_USERNAME": "testuser",
        }, clear=False):
            # Patch directory creation to use temp dir
            with patch.object(Path, 'mkdir'):
                config = Config()
                config.__post_init__()
                assert config.datagolf_api_key == "test_key"
                assert config.site_email == "test@test.com"
                assert config.site_password == "testpass"
                assert config.site_username == "testuser"

    def test_config_no_hardcoded_credentials(self):
        """Test that no credentials are hardcoded as defaults."""
        with patch.dict(os.environ, {
            "DATAGOLF_API_KEY": "",
            "PGA_OAD_EMAIL": "",
            "PGA_OAD_PASSWORD": "",
            "PGA_OAD_USERNAME": "",
        }, clear=False):
            with patch.object(Path, 'mkdir'):
                config = Config()
                config.__post_init__()
                # Verify no hardcoded values - should be empty strings
                assert config.site_email == ""
                assert config.site_password == ""
                assert config.site_username == ""
                # Specifically check that old hardcoded values are gone
                assert "gitberge" not in config.site_email
                assert "Sixers" not in config.site_password

    def test_validate_config_missing_api_key(self):
        """Test validate_config returns error when API key missing."""
        with patch.dict(os.environ, {
            "DATAGOLF_API_KEY": "",
        }, clear=False):
            with patch.object(Path, 'mkdir'):
                config = Config()
                config.__post_init__()
                errors = config.validate_config(require_api_key=True)
                assert len(errors) == 1
                assert "DATAGOLF_API_KEY" in errors[0]

    def test_validate_config_with_api_key(self):
        """Test validate_config returns no errors when API key present."""
        with patch.dict(os.environ, {
            "DATAGOLF_API_KEY": "valid_key",
        }, clear=False):
            with patch.object(Path, 'mkdir'):
                config = Config()
                config.__post_init__()
                errors = config.validate_config(require_api_key=True)
                assert len(errors) == 0

    def test_validate_config_api_key_not_required(self):
        """Test validate_config skips API key check when not required."""
        with patch.dict(os.environ, {
            "DATAGOLF_API_KEY": "",
        }, clear=False):
            with patch.object(Path, 'mkdir'):
                config = Config()
                config.__post_init__()
                errors = config.validate_config(require_api_key=False)
                assert len(errors) == 0

    def test_is_configured_true(self):
        """Test is_configured returns True when API key present."""
        with patch.dict(os.environ, {
            "DATAGOLF_API_KEY": "valid_key",
        }, clear=False):
            with patch.object(Path, 'mkdir'):
                config = Config()
                config.__post_init__()
                assert config.is_configured() is True

    def test_is_configured_false(self):
        """Test is_configured returns False when API key missing."""
        with patch.dict(os.environ, {
            "DATAGOLF_API_KEY": "",
        }, clear=False):
            with patch.object(Path, 'mkdir'):
                config = Config()
                config.__post_init__()
                assert config.is_configured() is False

    def test_directory_creation_permission_error(self, temp_dir):
        """Test that permission errors are handled properly."""
        with patch.dict(os.environ, {"DATAGOLF_API_KEY": ""}, clear=False):
            with patch.object(Path, 'mkdir', side_effect=PermissionError("Access denied")):
                with pytest.raises(RuntimeError) as exc_info:
                    config = Config()
                    config.__post_init__()
                assert "Permission denied" in str(exc_info.value)

    def test_directory_creation_os_error(self, temp_dir):
        """Test that OS errors are handled properly."""
        with patch.dict(os.environ, {"DATAGOLF_API_KEY": ""}, clear=False):
            with patch.object(Path, 'mkdir', side_effect=OSError("Disk error")):
                with pytest.raises(RuntimeError) as exc_info:
                    config = Config()
                    config.__post_init__()
                assert "Disk error" in str(exc_info.value)


class TestScheduleFunctions:
    """Tests for schedule-related functions."""

    def test_get_schedule_returns_list(self):
        """Test that get_schedule returns a list of tournaments."""
        schedule = get_schedule()
        assert isinstance(schedule, list)
        assert len(schedule) > 0

    def test_get_tournament_by_name_found(self):
        """Test finding a tournament by name."""
        tournament = get_tournament_by_name("Masters")
        assert tournament is not None
        assert "Masters" in tournament.name

    def test_get_tournament_by_name_not_found(self):
        """Test that None is returned for unknown tournament."""
        tournament = get_tournament_by_name("Nonexistent Tournament XYZ")
        assert tournament is None

    def test_get_tournament_by_name_case_insensitive(self):
        """Test that tournament search is case-insensitive."""
        tournament = get_tournament_by_name("MASTERS")
        assert tournament is not None
        tournament2 = get_tournament_by_name("masters")
        assert tournament2 is not None

    def test_schedule_has_majors(self):
        """Test that schedule includes major tournaments."""
        schedule = get_schedule()
        majors = [t for t in schedule if t.is_major]
        assert len(majors) == 4  # Masters, PGA, US Open, Open Championship

    def test_schedule_purses_are_positive(self):
        """Test that all tournament purses are positive."""
        schedule = get_schedule()
        for tournament in schedule:
            assert tournament.purse > 0
