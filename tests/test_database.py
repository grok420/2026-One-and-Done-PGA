"""
Tests for database.py - Database operations.
"""

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from database import Database, DatabaseError
from models import Golfer, GolferStats, Tournament, Pick, LeagueStanding, Tier


class TestDatabaseInitialization:
    """Tests for database initialization."""

    def test_database_creates_file(self, temp_db_path):
        """Test that database file is created."""
        with patch('database.get_config') as mock_config:
            mock_config.return_value.db_path = temp_db_path
            db = Database(db_path=temp_db_path)
            assert temp_db_path.exists()

    def test_database_creates_tables(self, temp_db_path):
        """Test that all required tables are created."""
        with patch('database.get_config') as mock_config:
            mock_config.return_value.db_path = temp_db_path
            db = Database(db_path=temp_db_path)

            conn = sqlite3.connect(temp_db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
            conn.close()

            expected_tables = {
                'tournaments', 'golfers', 'picks', 'opponent_picks',
                'standings', 'simulations', 'cache', 'golfer_probabilities',
                'available_golfers'
            }
            assert expected_tables.issubset(tables)

    def test_database_permission_error(self, temp_dir):
        """Test DatabaseError raised on permission denied."""
        with patch('database.get_config') as mock_config:
            mock_config.return_value.db_path = temp_dir / "test.db"
            with patch.object(Database, '_init_db', side_effect=sqlite3.OperationalError("permission denied")):
                with pytest.raises(DatabaseError) as exc_info:
                    Database(db_path=temp_dir / "test.db")
                assert "Permission denied" in str(exc_info.value)

    def test_database_disk_full_error(self, temp_dir):
        """Test DatabaseError raised on disk full."""
        with patch('database.get_config') as mock_config:
            mock_config.return_value.db_path = temp_dir / "test.db"
            with patch.object(Database, '_init_db', side_effect=sqlite3.OperationalError("disk full")):
                with pytest.raises(DatabaseError) as exc_info:
                    Database(db_path=temp_dir / "test.db")
                assert "Disk full" in str(exc_info.value)

    def test_database_unable_to_open_error(self, temp_dir):
        """Test DatabaseError raised when unable to open database."""
        with patch('database.get_config') as mock_config:
            mock_config.return_value.db_path = temp_dir / "test.db"
            with patch.object(Database, '_init_db', side_effect=sqlite3.OperationalError("unable to open")):
                with pytest.raises(DatabaseError) as exc_info:
                    Database(db_path=temp_dir / "test.db")
                assert "Cannot open database" in str(exc_info.value)


class TestGolferOperations:
    """Tests for golfer CRUD operations."""

    @pytest.fixture
    def db(self, temp_db_path):
        """Create a test database."""
        with patch('database.get_config') as mock_config:
            mock_config.return_value.db_path = temp_db_path
            return Database(db_path=temp_db_path)

    def test_save_and_get_golfer(self, db):
        """Test saving and retrieving a golfer."""
        golfer = Golfer(
            name="Test Golfer",
            owgr=10,
            datagolf_id="12345",
            stats=GolferStats(sg_total=1.5, sg_approach=0.8),
        )
        db.save_golfer(golfer)

        retrieved = db.get_golfer("Test Golfer")
        assert retrieved is not None
        assert retrieved.name == "Test Golfer"
        assert retrieved.owgr == 10
        assert retrieved.stats.sg_total == 1.5

    def test_get_golfer_not_found(self, db):
        """Test that None is returned for unknown golfer."""
        result = db.get_golfer("Unknown Golfer")
        assert result is None

    def test_get_all_golfers(self, db):
        """Test retrieving all golfers."""
        golfer1 = Golfer(name="Golfer A", owgr=1)
        golfer2 = Golfer(name="Golfer B", owgr=2)
        db.save_golfer(golfer1)
        db.save_golfer(golfer2)

        all_golfers = db.get_all_golfers()
        assert len(all_golfers) == 2
        # Should be ordered by OWGR
        assert all_golfers[0].owgr <= all_golfers[1].owgr

    def test_golfer_json_parsing_error_handling(self, db, temp_db_path):
        """Test that corrupted JSON in golfer stats is handled gracefully."""
        # Insert a golfer with corrupted JSON directly
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO golfers (name, owgr, stats_json, course_history_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, ("Corrupted Golfer", 50, "invalid json {{{", "also invalid", datetime.now().isoformat()))
        conn.commit()
        conn.close()

        # Should not raise an error
        golfer = db.get_golfer("Corrupted Golfer")
        assert golfer is not None
        assert golfer.name == "Corrupted Golfer"
        # Stats should be empty/default due to JSON error
        assert golfer.stats.sg_total == 0


class TestPickOperations:
    """Tests for pick CRUD operations."""

    @pytest.fixture
    def db(self, temp_db_path):
        """Create a test database."""
        with patch('database.get_config') as mock_config:
            mock_config.return_value.db_path = temp_db_path
            return Database(db_path=temp_db_path)

    def test_save_and_get_pick(self, db):
        """Test saving and retrieving a pick."""
        pick = Pick(
            golfer_name="Scottie Scheffler",
            tournament_name="The Masters",
            tournament_date=date(2026, 4, 9),
            earnings=3600000,
            position=1,
            made_cut=True,
            is_major=True,
        )
        db.save_pick(pick)

        picks = db.get_all_picks()
        assert len(picks) == 1
        assert picks[0].golfer_name == "Scottie Scheffler"
        assert picks[0].earnings == 3600000

    def test_get_used_golfers(self, db):
        """Test getting list of used golfers."""
        pick1 = Pick(golfer_name="Golfer A", tournament_name="T1", tournament_date=date(2026, 1, 1))
        pick2 = Pick(golfer_name="Golfer B", tournament_name="T2", tournament_date=date(2026, 1, 8))
        db.save_pick(pick1)
        db.save_pick(pick2)

        used = db.get_used_golfers()
        assert "Golfer A" in used
        assert "Golfer B" in used


class TestCacheOperations:
    """Tests for cache operations."""

    @pytest.fixture
    def db(self, temp_db_path):
        """Create a test database."""
        with patch('database.get_config') as mock_config:
            mock_config.return_value.db_path = temp_db_path
            return Database(db_path=temp_db_path)

    def test_set_and_get_cache(self, db):
        """Test setting and getting cache entries."""
        test_data = {"key": "value", "nested": {"a": 1}}
        expires = datetime.now() + timedelta(hours=1)

        db.set_cache("test_key", test_data, expires)
        result = db.get_cache("test_key")

        assert result == test_data

    def test_cache_expiration(self, db):
        """Test that expired cache entries return None."""
        test_data = {"key": "value"}
        expires = datetime.now() - timedelta(hours=1)  # Already expired

        db.set_cache("expired_key", test_data, expires)
        result = db.get_cache("expired_key")

        assert result is None

    def test_cache_json_error_handling(self, db, temp_db_path):
        """Test that corrupted cache JSON is handled gracefully."""
        # Insert corrupted cache entry directly
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        expires = (datetime.now() + timedelta(hours=1)).isoformat()
        cursor.execute(
            "INSERT INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
            ("corrupted_cache", "not valid json {{{", expires)
        )
        conn.commit()
        conn.close()

        # Should return None and not raise an error
        result = db.get_cache("corrupted_cache")
        assert result is None

    def test_clear_expired_cache(self, db):
        """Test clearing expired cache entries."""
        # Add one valid and one expired entry
        valid_expires = datetime.now() + timedelta(hours=1)
        expired_expires = datetime.now() - timedelta(hours=1)

        db.set_cache("valid_key", {"data": 1}, valid_expires)
        db.set_cache("expired_key", {"data": 2}, expired_expires)

        db.clear_expired_cache()

        assert db.get_cache("valid_key") is not None
        # Expired entry should already be gone after get_cache or clear


class TestTournamentOperations:
    """Tests for tournament CRUD operations."""

    @pytest.fixture
    def db(self, temp_db_path):
        """Create a test database."""
        with patch('database.get_config') as mock_config:
            mock_config.return_value.db_path = temp_db_path
            return Database(db_path=temp_db_path)

    def test_save_and_get_tournament(self, db):
        """Test saving and retrieving a tournament."""
        tournament = Tournament(
            name="Test Tournament",
            date=date(2026, 5, 1),
            purse=10000000,
            course="Test Course",
            tier=Tier.TIER_1,
            is_major=True,
        )
        db.save_tournament(tournament)

        retrieved = db.get_tournament("Test Tournament")
        assert retrieved is not None
        assert retrieved.name == "Test Tournament"
        assert retrieved.purse == 10000000
        assert retrieved.is_major is True

    def test_get_all_tournaments(self, db):
        """Test retrieving all tournaments."""
        t1 = Tournament(name="Tournament A", date=date(2026, 1, 1), purse=5000000)
        t2 = Tournament(name="Tournament B", date=date(2026, 2, 1), purse=6000000)
        db.save_tournament(t1)
        db.save_tournament(t2)

        all_tournaments = db.get_all_tournaments()
        assert len(all_tournaments) == 2


class TestStandingsOperations:
    """Tests for standings operations."""

    @pytest.fixture
    def db(self, temp_db_path):
        """Create a test database."""
        with patch('database.get_config') as mock_config:
            mock_config.return_value.db_path = temp_db_path
            return Database(db_path=temp_db_path)

    def test_save_and_get_standings(self, db):
        """Test saving and retrieving standings."""
        standings = [
            LeagueStanding(rank=1, player_name="Player One", username="player1", total_earnings=500000),
            LeagueStanding(rank=2, player_name="Player Two", username="player2", total_earnings=400000),
        ]
        db.save_standings(standings)

        retrieved = db.get_latest_standings()
        assert len(retrieved) == 2
        assert retrieved[0].rank == 1
        assert retrieved[0].username == "player1"

    def test_get_my_standing(self, db):
        """Test getting user's own standing."""
        standings = [
            LeagueStanding(rank=1, player_name="Player One", username="player1", total_earnings=500000),
            LeagueStanding(rank=2, player_name="Me", username="myuser", total_earnings=400000),
        ]
        db.save_standings(standings)

        my_standing = db.get_my_standing("myuser")
        assert my_standing is not None
        assert my_standing.rank == 2

    def test_get_my_standing_not_found(self, db):
        """Test that None is returned if user not in standings."""
        standings = [
            LeagueStanding(rank=1, player_name="Player One", username="player1", total_earnings=500000),
        ]
        db.save_standings(standings)

        my_standing = db.get_my_standing("unknown_user")
        assert my_standing is None


class TestUtilityOperations:
    """Tests for utility database operations."""

    @pytest.fixture
    def db(self, temp_db_path):
        """Create a test database."""
        with patch('database.get_config') as mock_config:
            mock_config.return_value.db_path = temp_db_path
            return Database(db_path=temp_db_path)

    def test_get_total_earnings(self, db):
        """Test calculating total earnings."""
        pick1 = Pick(golfer_name="G1", tournament_name="T1", tournament_date=date(2026, 1, 1), earnings=100000)
        pick2 = Pick(golfer_name="G2", tournament_name="T2", tournament_date=date(2026, 1, 8), earnings=200000)
        db.save_pick(pick1)
        db.save_pick(pick2)

        total = db.get_total_earnings()
        assert total == 300000

    def test_get_total_earnings_empty(self, db):
        """Test total earnings with no picks."""
        total = db.get_total_earnings()
        assert total == 0

    def test_get_cuts_made_count(self, db):
        """Test counting cuts made."""
        pick1 = Pick(golfer_name="G1", tournament_name="T1", tournament_date=date(2026, 1, 1), made_cut=True)
        pick2 = Pick(golfer_name="G2", tournament_name="T2", tournament_date=date(2026, 1, 8), made_cut=False)
        pick3 = Pick(golfer_name="G3", tournament_name="T3", tournament_date=date(2026, 1, 15), made_cut=True)
        db.save_pick(pick1)
        db.save_pick(pick2)
        db.save_pick(pick3)

        cuts = db.get_cuts_made_count()
        assert cuts == 2

    def test_get_picks_count(self, db):
        """Test counting total picks."""
        pick1 = Pick(golfer_name="G1", tournament_name="T1", tournament_date=date(2026, 1, 1))
        pick2 = Pick(golfer_name="G2", tournament_name="T2", tournament_date=date(2026, 1, 8))
        db.save_pick(pick1)
        db.save_pick(pick2)

        count = db.get_picks_count()
        assert count == 2
