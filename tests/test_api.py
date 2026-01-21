"""
Tests for api.py - Data Golf API client.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock

import pytest
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from api import DataGolfAPI, PredictionData


class TestDataGolfAPIInitialization:
    """Tests for API client initialization."""

    def test_api_uses_config_key(self, temp_db_path):
        """Test that API uses key from config."""
        with patch('api.get_config') as mock_config:
            mock_config.return_value.datagolf_api_key = "config_key"
            with patch('api.Database'):
                api = DataGolfAPI()
                assert api.api_key == "config_key"

    def test_api_uses_provided_key(self, temp_db_path):
        """Test that provided key overrides config."""
        with patch('api.get_config') as mock_config:
            mock_config.return_value.datagolf_api_key = "config_key"
            with patch('api.Database'):
                api = DataGolfAPI(api_key="provided_key")
                assert api.api_key == "provided_key"


class TestAPIKeyValidation:
    """Tests for API key validation."""

    def test_request_raises_error_without_api_key(self, temp_db_path):
        """Test that _request raises ValueError when API key is missing."""
        with patch('api.get_config') as mock_config:
            mock_config.return_value.datagolf_api_key = ""
            with patch('api.Database'):
                api = DataGolfAPI()

                with pytest.raises(ValueError) as exc_info:
                    api._request("/test-endpoint")

                assert "DATAGOLF_API_KEY not configured" in str(exc_info.value)
                assert "datagolf.com/api-access" in str(exc_info.value)

    def test_request_proceeds_with_api_key(self, temp_db_path):
        """Test that _request proceeds when API key is present."""
        with patch('api.get_config') as mock_config:
            mock_config.return_value.datagolf_api_key = "valid_key"
            mock_db = MagicMock()
            mock_db.get_cache.return_value = {"cached": "data"}

            with patch('api.Database', return_value=mock_db):
                api = DataGolfAPI()
                result = api._request("/test-endpoint")

                # Should return cached data
                assert result == {"cached": "data"}


class TestAPIRetryLogic:
    """Tests for API retry logic with exponential backoff."""

    def test_retry_on_request_failure(self, temp_db_path):
        """Test that API retries on transient failures."""
        with patch('api.get_config') as mock_config:
            mock_config.return_value.datagolf_api_key = "valid_key"
            mock_db = MagicMock()
            mock_db.get_cache.return_value = None  # No cache

            with patch('api.Database', return_value=mock_db):
                api = DataGolfAPI()

                # Mock session to fail twice then succeed
                mock_response = MagicMock()
                mock_response.json.return_value = {"success": True}
                mock_response.raise_for_status = MagicMock()

                call_count = 0
                def side_effect(*args, **kwargs):
                    nonlocal call_count
                    call_count += 1
                    if call_count < 3:
                        raise requests.RequestException("Connection failed")
                    return mock_response

                api._session.get = MagicMock(side_effect=side_effect)

                with patch('time.sleep'):  # Don't actually sleep
                    result = api._request("/test-endpoint")

                assert result == {"success": True}
                assert call_count == 3

    def test_gives_up_after_max_retries(self, temp_db_path):
        """Test that API gives up after max retry attempts."""
        with patch('api.get_config') as mock_config:
            mock_config.return_value.datagolf_api_key = "valid_key"
            mock_db = MagicMock()
            mock_db.get_cache.return_value = None

            with patch('api.Database', return_value=mock_db):
                api = DataGolfAPI()
                api._session.get = MagicMock(side_effect=requests.RequestException("Always fails"))

                with patch('time.sleep'):
                    result = api._request("/test-endpoint")

                assert result is None
                assert api._session.get.call_count == 3  # max_retries

    def test_exponential_backoff_delays(self, temp_db_path):
        """Test that retry uses exponential backoff."""
        with patch('api.get_config') as mock_config:
            mock_config.return_value.datagolf_api_key = "valid_key"
            mock_db = MagicMock()
            mock_db.get_cache.return_value = None

            with patch('api.Database', return_value=mock_db):
                api = DataGolfAPI()
                api._session.get = MagicMock(side_effect=requests.RequestException("Fails"))

                sleep_calls = []
                with patch('time.sleep', side_effect=lambda x: sleep_calls.append(x)):
                    api._request("/test-endpoint")

                # Should have delays of 1.0, 2.0 (base_delay * 2^attempt)
                assert len(sleep_calls) == 2
                assert sleep_calls[0] == 1.0  # 1.0 * 2^0
                assert sleep_calls[1] == 2.0  # 1.0 * 2^1


class TestJSONErrorHandling:
    """Tests for JSON parsing error handling."""

    def test_handles_json_decode_error(self, temp_db_path):
        """Test that JSON decode errors are handled gracefully."""
        with patch('api.get_config') as mock_config:
            mock_config.return_value.datagolf_api_key = "valid_key"
            mock_db = MagicMock()
            mock_db.get_cache.return_value = None

            with patch('api.Database', return_value=mock_db):
                api = DataGolfAPI()

                mock_response = MagicMock()
                mock_response.raise_for_status = MagicMock()
                mock_response.json.side_effect = json.JSONDecodeError("error", "doc", 0)
                api._session.get = MagicMock(return_value=mock_response)

                result = api._request("/test-endpoint")

                assert result is None


class TestAPICaching:
    """Tests for API response caching."""

    def test_returns_cached_data(self, temp_db_path):
        """Test that cached data is returned when available."""
        with patch('api.get_config') as mock_config:
            mock_config.return_value.datagolf_api_key = "valid_key"
            mock_db = MagicMock()
            mock_db.get_cache.return_value = {"cached": "response"}

            with patch('api.Database', return_value=mock_db):
                api = DataGolfAPI()
                # Mock the session to verify it's not called
                api._session = MagicMock()
                result = api._request("/test-endpoint")

                assert result == {"cached": "response"}
                # Session should not be called when cache hit
                api._session.get.assert_not_called()

    def test_caches_api_response(self, temp_db_path):
        """Test that API responses are cached."""
        with patch('api.get_config') as mock_config:
            mock_config.return_value.datagolf_api_key = "valid_key"
            mock_db = MagicMock()
            mock_db.get_cache.return_value = None  # No cache

            with patch('api.Database', return_value=mock_db):
                api = DataGolfAPI()

                mock_response = MagicMock()
                mock_response.json.return_value = {"fresh": "data"}
                mock_response.raise_for_status = MagicMock()
                api._session.get = MagicMock(return_value=mock_response)

                result = api._request("/test-endpoint", cache_hours=2)

                # Verify cache was set
                mock_db.set_cache.assert_called_once()
                call_args = mock_db.set_cache.call_args
                assert call_args[0][1] == {"fresh": "data"}


class TestHealthCheck:
    """Tests for API health check."""

    def test_health_check_returns_false_without_key(self, temp_db_path):
        """Test health check returns False when API key missing."""
        with patch('api.get_config') as mock_config:
            mock_config.return_value.datagolf_api_key = ""
            with patch('api.Database'):
                api = DataGolfAPI()
                assert api.health_check() is False

    def test_health_check_returns_true_on_success(self, temp_db_path):
        """Test health check returns True on successful API call."""
        with patch('api.get_config') as mock_config:
            mock_config.return_value.datagolf_api_key = "valid_key"
            with patch('api.Database'):
                api = DataGolfAPI()

                mock_response = MagicMock()
                mock_response.status_code = 200
                api._session.get = MagicMock(return_value=mock_response)

                assert api.health_check() is True

    def test_health_check_returns_false_on_failure(self, temp_db_path):
        """Test health check returns False on API failure."""
        with patch('api.get_config') as mock_config:
            mock_config.return_value.datagolf_api_key = "valid_key"
            with patch('api.Database'):
                api = DataGolfAPI()

                mock_response = MagicMock()
                mock_response.status_code = 401
                api._session.get = MagicMock(return_value=mock_response)

                assert api.health_check() is False

    def test_health_check_returns_false_on_exception(self, temp_db_path):
        """Test health check returns False on request exception."""
        with patch('api.get_config') as mock_config:
            mock_config.return_value.datagolf_api_key = "valid_key"
            with patch('api.Database'):
                api = DataGolfAPI()
                api._session.get = MagicMock(side_effect=requests.RequestException("Failed"))

                assert api.health_check() is False


class TestPredictionData:
    """Tests for prediction data parsing."""

    def test_get_pre_tournament_predictions(self, temp_db_path, sample_prediction_response):
        """Test parsing pre-tournament predictions."""
        with patch('api.get_config') as mock_config:
            mock_config.return_value.datagolf_api_key = "valid_key"
            mock_db = MagicMock()
            mock_db.get_cache.return_value = None

            with patch('api.Database', return_value=mock_db):
                api = DataGolfAPI()

                mock_response = MagicMock()
                mock_response.json.return_value = sample_prediction_response
                mock_response.raise_for_status = MagicMock()
                api._session.get = MagicMock(return_value=mock_response)

                predictions = api.get_pre_tournament_predictions()

                assert len(predictions) == 2
                assert predictions[0].golfer_name == "Scottie Scheffler"
                assert predictions[0].win_prob == 0.15
                assert predictions[1].golfer_name == "Rory McIlroy"

    def test_get_pre_tournament_predictions_empty_response(self, temp_db_path):
        """Test handling empty prediction response."""
        with patch('api.get_config') as mock_config:
            mock_config.return_value.datagolf_api_key = "valid_key"
            mock_db = MagicMock()
            mock_db.get_cache.return_value = None

            with patch('api.Database', return_value=mock_db):
                api = DataGolfAPI()

                mock_response = MagicMock()
                mock_response.json.return_value = {}
                mock_response.raise_for_status = MagicMock()
                api._session.get = MagicMock(return_value=mock_response)

                predictions = api.get_pre_tournament_predictions()

                assert predictions == []


class TestPlayerList:
    """Tests for player list retrieval."""

    def test_get_player_list(self, temp_db_path):
        """Test getting player list."""
        with patch('api.get_config') as mock_config:
            mock_config.return_value.datagolf_api_key = "valid_key"
            mock_db = MagicMock()
            mock_db.get_cache.return_value = None

            with patch('api.Database', return_value=mock_db):
                api = DataGolfAPI()

                mock_response = MagicMock()
                mock_response.json.return_value = [
                    {"player_name": "Player A", "dg_id": "1", "owgr": 1, "country": "USA"},
                    {"player_name": "Player B", "dg_id": "2", "owgr": 2, "country": "ENG"},
                ]
                mock_response.raise_for_status = MagicMock()
                api._session.get = MagicMock(return_value=mock_response)

                players = api.get_player_list()

                assert len(players) == 2
                assert players[0]["name"] == "Player A"
                assert players[0]["owgr"] == 1

    def test_get_player_list_empty(self, temp_db_path):
        """Test handling empty player list."""
        with patch('api.get_config') as mock_config:
            mock_config.return_value.datagolf_api_key = "valid_key"
            mock_db = MagicMock()
            mock_db.get_cache.return_value = None

            with patch('api.Database', return_value=mock_db):
                api = DataGolfAPI()

                # Simulate API returning None (failure)
                mock_response = MagicMock()
                mock_response.raise_for_status.side_effect = requests.HTTPError("404")
                api._session.get = MagicMock(return_value=mock_response)

                with patch('time.sleep'):
                    players = api.get_player_list()

                assert players == []
