"""
SQLite database layer for PGA One and Done Optimizer.
Handles persistence of picks, standings, golfer data, and simulations.
"""

import sqlite3
import json
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

try:
    from .models import (
        Tournament, Golfer, GolferStats, Pick, LeagueStanding,
        OpponentPick, SimulationResult, Tier, CourseHistory,
        GolferAvailability, SeasonPlanEntry, Entry
    )
    from .config import get_config
except ImportError:
    from models import (
        Tournament, Golfer, GolferStats, Pick, LeagueStanding,
        OpponentPick, SimulationResult, Tier, CourseHistory,
        GolferAvailability, SeasonPlanEntry, Entry
    )
    from config import get_config


class DatabaseError(Exception):
    """Custom exception for database errors."""
    pass


class Database:
    """SQLite database manager."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database connection."""
        config = get_config()
        self.db_path = db_path or config.db_path
        try:
            self._init_db()
        except sqlite3.OperationalError as e:
            error_msg = str(e).lower()
            if "permission denied" in error_msg or "read-only" in error_msg:
                raise DatabaseError(
                    f"Permission denied: Cannot create or access database at {self.db_path}. "
                    "Check file and directory permissions."
                ) from e
            elif "disk" in error_msg and "full" in error_msg:
                raise DatabaseError(
                    f"Disk full: Cannot create database at {self.db_path}. "
                    "Free up disk space and try again."
                ) from e
            elif "unable to open" in error_msg:
                raise DatabaseError(
                    f"Cannot open database at {self.db_path}. "
                    "Ensure the parent directory exists and is writable."
                ) from e
            else:
                raise DatabaseError(f"Database initialization failed: {e}") from e
        except Exception as e:
            raise DatabaseError(f"Unexpected error initializing database: {e}") from e

    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self):
        """Initialize database schema."""
        with self._connection() as conn:
            cursor = conn.cursor()

            # Tournaments table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tournaments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    date TEXT NOT NULL,
                    purse INTEGER NOT NULL,
                    course TEXT,
                    tier INTEGER,
                    is_major INTEGER DEFAULT 0,
                    is_signature INTEGER DEFAULT 0,
                    is_playoff INTEGER DEFAULT 0,
                    is_opposite_field INTEGER DEFAULT 0,
                    UNIQUE(name, date)
                )
            """)

            # Golfers table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS golfers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    owgr INTEGER DEFAULT 999,
                    datagolf_id TEXT,
                    stats_json TEXT,
                    course_history_json TEXT,
                    updated_at TEXT
                )
            """)

            # User picks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS picks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    golfer_name TEXT NOT NULL,
                    tournament_name TEXT NOT NULL,
                    tournament_date TEXT NOT NULL,
                    earnings INTEGER DEFAULT 0,
                    position INTEGER,
                    made_cut INTEGER DEFAULT 0,
                    is_major INTEGER DEFAULT 0,
                    created_at TEXT,
                    UNIQUE(tournament_name, tournament_date)
                )
            """)

            # Opponent picks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS opponent_picks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    opponent_username TEXT NOT NULL,
                    golfer_name TEXT NOT NULL,
                    tournament_name TEXT NOT NULL,
                    tournament_date TEXT NOT NULL,
                    created_at TEXT,
                    UNIQUE(opponent_username, tournament_name, tournament_date)
                )
            """)

            # League standings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS standings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_date TEXT NOT NULL,
                    rank INTEGER NOT NULL,
                    player_name TEXT NOT NULL,
                    username TEXT NOT NULL,
                    total_earnings INTEGER DEFAULT 0,
                    cuts_made INTEGER DEFAULT 0,
                    picks_made INTEGER DEFAULT 0,
                    majors_earnings INTEGER DEFAULT 0,
                    UNIQUE(snapshot_date, username)
                )
            """)

            # Simulation cache table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS simulations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    golfer_name TEXT NOT NULL,
                    tournament_name TEXT NOT NULL,
                    n_simulations INTEGER,
                    mean_earnings REAL,
                    median_earnings REAL,
                    std_earnings REAL,
                    percentile_10 REAL,
                    percentile_25 REAL,
                    percentile_75 REAL,
                    percentile_90 REAL,
                    win_count INTEGER,
                    top_10_count INTEGER,
                    cut_made_count INTEGER,
                    created_at TEXT,
                    UNIQUE(golfer_name, tournament_name)
                )
            """)

            # Cache table for API/scraper data
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
            """)

            # Golfer probabilities (from API)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS golfer_probabilities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    golfer_name TEXT NOT NULL,
                    tournament_name TEXT NOT NULL,
                    win_prob REAL DEFAULT 0,
                    top_5_prob REAL DEFAULT 0,
                    top_10_prob REAL DEFAULT 0,
                    top_20_prob REAL DEFAULT 0,
                    make_cut_prob REAL DEFAULT 0,
                    updated_at TEXT,
                    UNIQUE(golfer_name, tournament_name)
                )
            """)

            # Available golfers (not yet used)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS available_golfers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    golfer_name TEXT NOT NULL UNIQUE,
                    updated_at TEXT
                )
            """)

            # Course history (Phase 2.1)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS course_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    golfer_name TEXT NOT NULL,
                    course_name TEXT NOT NULL,
                    tournament_name TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    finish_position INTEGER,
                    earnings INTEGER DEFAULT 0,
                    sg_total REAL DEFAULT 0,
                    made_cut INTEGER DEFAULT 1,
                    created_at TEXT,
                    UNIQUE(golfer_name, course_name, year)
                )
            """)

            # Golfer availability (Phase 2.2)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS golfer_availability (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    golfer_name TEXT NOT NULL,
                    tournament_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    probability REAL DEFAULT 0,
                    updated_at TEXT,
                    UNIQUE(golfer_name, tournament_name)
                )
            """)

            # Season plan (Phase 2.3)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS season_plan (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id INTEGER DEFAULT 1,
                    tournament_name TEXT NOT NULL,
                    tournament_date TEXT NOT NULL,
                    golfer_name TEXT,
                    is_tentative INTEGER DEFAULT 1,
                    projected_ev REAL DEFAULT 0,
                    notes TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    UNIQUE(entry_id, tournament_name)
                )
            """)

            # Multi-entry support (Phase 3.2)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_name TEXT NOT NULL UNIQUE,
                    created_at TEXT
                )
            """)

            # Entry picks - separate pick tracking per entry (Phase 3.2)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS entry_picks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entry_id INTEGER NOT NULL,
                    golfer_name TEXT NOT NULL,
                    tournament_name TEXT NOT NULL,
                    tournament_date TEXT NOT NULL,
                    earnings INTEGER DEFAULT 0,
                    position INTEGER,
                    made_cut INTEGER DEFAULT 0,
                    created_at TEXT,
                    UNIQUE(entry_id, tournament_name),
                    FOREIGN KEY (entry_id) REFERENCES entries(id)
                )
            """)

            # =========================================================================
            # Learning Feature Tables
            # =========================================================================

            # Pick outcomes - Track predictions vs actual results for learning
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pick_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    golfer_name TEXT NOT NULL,
                    tournament_name TEXT NOT NULL,
                    tournament_date TEXT,
                    purse INTEGER,
                    predicted_win_prob REAL,
                    predicted_top10_prob REAL,
                    predicted_ev REAL,
                    predicted_course_fit REAL,
                    strategic_score REAL,
                    was_save_warning INTEGER DEFAULT 0,
                    actual_position INTEGER,
                    actual_earnings INTEGER,
                    made_cut INTEGER,
                    was_my_pick INTEGER DEFAULT 0,
                    outcome_recorded INTEGER DEFAULT 0,
                    created_at TEXT,
                    outcome_recorded_at TEXT,
                    UNIQUE(golfer_name, tournament_name, tournament_date)
                )
            """)

            # Learned course fits - Calibrated weights based on real outcomes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learned_course_fits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tournament_name TEXT NOT NULL,
                    skill_name TEXT NOT NULL,
                    static_weight REAL,
                    learned_weight REAL,
                    confidence REAL DEFAULT 0.5,
                    sample_size INTEGER DEFAULT 0,
                    last_updated TEXT,
                    UNIQUE(tournament_name, skill_name)
                )
            """)

            # Learned elite tiers - Dynamic tier adjustments based on performance
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learned_elite_tiers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    golfer_name TEXT NOT NULL UNIQUE,
                    static_tier INTEGER,
                    learned_tier INTEGER,
                    performance_score REAL,
                    tier_confidence REAL DEFAULT 0.5,
                    wins_this_season INTEGER DEFAULT 0,
                    top10s_this_season INTEGER DEFAULT 0,
                    events_played INTEGER DEFAULT 0,
                    total_earnings REAL DEFAULT 0,
                    last_updated TEXT
                )
            """)

            # Predictions - Store predictions for accuracy tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tournament_name TEXT NOT NULL,
                    golfer_name TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    predicted_value REAL NOT NULL,
                    actual_value REAL,
                    error REAL,
                    squared_error REAL,
                    prediction_made_at TEXT,
                    outcome_recorded_at TEXT,
                    UNIQUE(tournament_name, golfer_name, prediction_type)
                )
            """)

            # Model accuracy - Aggregate accuracy metrics over time
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_accuracy (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    tournament_name TEXT,
                    sample_size INTEGER,
                    time_period TEXT,
                    recorded_at TEXT
                )
            """)

            # Opponent patterns - Learn opponent picking behavior
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS opponent_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    opponent_name TEXT NOT NULL UNIQUE,
                    prefers_favorites REAL DEFAULT 0.5,
                    prefers_value REAL DEFAULT 0.5,
                    prefers_course_fit REAL DEFAULT 0.5,
                    risk_tolerance REAL DEFAULT 0.5,
                    avg_golfer_ranking REAL,
                    avg_win_prob_selected REAL,
                    total_picks_tracked INTEGER DEFAULT 0,
                    last_updated TEXT
                )
            """)

    # =========================================================================
    # Tournament operations
    # =========================================================================

    def save_tournament(self, tournament: Tournament) -> int:
        """Save or update a tournament."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO tournaments
                (name, date, purse, course, tier, is_major, is_signature, is_playoff, is_opposite_field)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                tournament.name,
                tournament.date.isoformat(),
                tournament.purse,
                tournament.course,
                tournament.tier.value,
                1 if tournament.is_major else 0,
                1 if tournament.is_signature else 0,
                1 if tournament.is_playoff else 0,
                1 if tournament.is_opposite_field else 0,
            ))
            return cursor.lastrowid

    def get_tournament(self, name: str) -> Optional[Tournament]:
        """Get tournament by name."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tournaments WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                return self._row_to_tournament(row)
        return None

    def get_all_tournaments(self) -> List[Tournament]:
        """Get all tournaments."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tournaments ORDER BY date")
            return [self._row_to_tournament(row) for row in cursor.fetchall()]

    def _row_to_tournament(self, row: sqlite3.Row) -> Tournament:
        """Convert database row to Tournament object."""
        return Tournament(
            name=row["name"],
            date=date.fromisoformat(row["date"]),
            purse=row["purse"],
            course=row["course"] or "",
            tier=Tier(row["tier"]) if row["tier"] else Tier.TIER_2,
            is_major=bool(row["is_major"]),
            is_signature=bool(row["is_signature"]),
            is_playoff=bool(row["is_playoff"]),
            is_opposite_field=bool(row["is_opposite_field"]),
        )

    # =========================================================================
    # Golfer operations
    # =========================================================================

    def save_golfer(self, golfer: Golfer) -> int:
        """Save or update a golfer."""
        with self._connection() as conn:
            cursor = conn.cursor()
            stats_json = json.dumps({
                "sg_total": golfer.stats.sg_total,
                "sg_off_tee": golfer.stats.sg_off_tee,
                "sg_approach": golfer.stats.sg_approach,
                "sg_around_green": golfer.stats.sg_around_green,
                "sg_putting": golfer.stats.sg_putting,
                "driving_distance": golfer.stats.driving_distance,
                "driving_accuracy": golfer.stats.driving_accuracy,
                "gir_pct": golfer.stats.gir_pct,
                "scrambling_pct": golfer.stats.scrambling_pct,
            })
            cursor.execute("""
                INSERT OR REPLACE INTO golfers
                (name, owgr, datagolf_id, stats_json, course_history_json, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                golfer.name,
                golfer.owgr,
                golfer.datagolf_id,
                stats_json,
                json.dumps(golfer.course_history),
                datetime.now().isoformat(),
            ))
            return cursor.lastrowid

    def get_golfer(self, name: str) -> Optional[Golfer]:
        """Get golfer by name."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM golfers WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                return self._row_to_golfer(row)
        return None

    def get_all_golfers(self) -> List[Golfer]:
        """Get all golfers."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM golfers ORDER BY owgr")
            return [self._row_to_golfer(row) for row in cursor.fetchall()]

    def _row_to_golfer(self, row: sqlite3.Row) -> Golfer:
        """Convert database row to Golfer object."""
        stats_data = {}
        course_history = {}
        if row["stats_json"]:
            try:
                stats_data = json.loads(row["stats_json"])
            except json.JSONDecodeError:
                pass
        if row["course_history_json"]:
            try:
                course_history = json.loads(row["course_history_json"])
            except json.JSONDecodeError:
                pass
        return Golfer(
            name=row["name"],
            owgr=row["owgr"],
            datagolf_id=row["datagolf_id"],
            stats=GolferStats(**stats_data),
            course_history=course_history,
        )

    # =========================================================================
    # Pick operations
    # =========================================================================

    def save_pick(self, pick: Pick) -> int:
        """Save or update a pick."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO picks
                (golfer_name, tournament_name, tournament_date, earnings, position, made_cut, is_major, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pick.golfer_name,
                pick.tournament_name,
                pick.tournament_date.isoformat(),
                pick.earnings,
                pick.position,
                1 if pick.made_cut else 0,
                1 if pick.is_major else 0,
                datetime.now().isoformat(),
            ))
            return cursor.lastrowid

    def get_all_picks(self) -> List[Pick]:
        """Get all user picks."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM picks ORDER BY tournament_date")
            return [self._row_to_pick(row) for row in cursor.fetchall()]

    def get_used_golfers(self) -> List[str]:
        """Get list of golfer names already used."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT golfer_name FROM picks")
            return [row["golfer_name"] for row in cursor.fetchall()]

    def _row_to_pick(self, row: sqlite3.Row) -> Pick:
        """Convert database row to Pick object."""
        return Pick(
            golfer_name=row["golfer_name"],
            tournament_name=row["tournament_name"],
            tournament_date=date.fromisoformat(row["tournament_date"]),
            earnings=row["earnings"],
            position=row["position"],
            made_cut=bool(row["made_cut"]),
            is_major=bool(row["is_major"]),
        )

    # =========================================================================
    # Opponent pick operations
    # =========================================================================

    def save_opponent_pick(self, pick: OpponentPick) -> int:
        """Save an opponent's pick."""
        with self._connection() as conn:
            cursor = conn.cursor()
            # Handle both date objects and string dates
            t_date = pick.tournament_date
            if hasattr(t_date, 'isoformat'):
                t_date = t_date.isoformat()
            cursor.execute("""
                INSERT OR REPLACE INTO opponent_picks
                (opponent_username, golfer_name, tournament_name, tournament_date, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                pick.opponent_username,
                pick.golfer_name,
                pick.tournament_name,
                t_date,
                datetime.now().isoformat(),
            ))
            return cursor.lastrowid

    def get_opponent_picks(self, opponent: Optional[str] = None) -> List[OpponentPick]:
        """Get opponent picks, optionally filtered by opponent."""
        with self._connection() as conn:
            cursor = conn.cursor()
            if opponent:
                cursor.execute(
                    "SELECT * FROM opponent_picks WHERE opponent_username = ? ORDER BY tournament_date",
                    (opponent,)
                )
            else:
                cursor.execute("SELECT * FROM opponent_picks ORDER BY tournament_date")
            return [
                OpponentPick(
                    opponent_username=row["opponent_username"],
                    golfer_name=row["golfer_name"],
                    tournament_name=row["tournament_name"],
                    tournament_date=date.fromisoformat(row["tournament_date"]),
                )
                for row in cursor.fetchall()
            ]

    def get_golfer_usage_count(self, golfer_name: str) -> int:
        """Get how many opponents have used a golfer."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(DISTINCT opponent_username) FROM opponent_picks WHERE golfer_name = ?",
                (golfer_name,)
            )
            return cursor.fetchone()[0]

    def get_all_golfer_usage(self) -> Dict[str, int]:
        """Get usage counts for all golfers."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT golfer_name, COUNT(DISTINCT opponent_username) as usage_count
                FROM opponent_picks
                GROUP BY golfer_name
                ORDER BY usage_count DESC
            """)
            return {row["golfer_name"]: row["usage_count"] for row in cursor.fetchall()}

    # =========================================================================
    # Standings operations
    # =========================================================================

    def save_standings(self, standings: List[LeagueStanding], snapshot_date: Optional[date] = None):
        """Save league standings snapshot."""
        snapshot_date = snapshot_date or date.today()
        with self._connection() as conn:
            cursor = conn.cursor()
            for standing in standings:
                cursor.execute("""
                    INSERT OR REPLACE INTO standings
                    (snapshot_date, rank, player_name, username, total_earnings, cuts_made, picks_made, majors_earnings)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot_date.isoformat(),
                    standing.rank,
                    standing.player_name,
                    standing.username,
                    standing.total_earnings,
                    standing.cuts_made,
                    standing.picks_made,
                    standing.majors_earnings,
                ))

    def get_latest_standings(self) -> List[LeagueStanding]:
        """Get most recent standings snapshot."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT MAX(snapshot_date) FROM standings")
            latest = cursor.fetchone()[0]
            if not latest:
                return []
            cursor.execute(
                "SELECT * FROM standings WHERE snapshot_date = ? ORDER BY rank",
                (latest,)
            )
            return [
                LeagueStanding(
                    rank=row["rank"],
                    player_name=row["player_name"],
                    username=row["username"],
                    total_earnings=row["total_earnings"],
                    cuts_made=row["cuts_made"],
                    picks_made=row["picks_made"],
                    majors_earnings=row["majors_earnings"],
                )
                for row in cursor.fetchall()
            ]

    def get_my_standing(self, username: str) -> Optional[LeagueStanding]:
        """Get user's current standing."""
        standings = self.get_latest_standings()
        for s in standings:
            if s.username.lower() == username.lower():
                return s
        return None

    # =========================================================================
    # Simulation cache operations
    # =========================================================================

    def save_simulation(self, result: SimulationResult) -> int:
        """Save simulation result."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO simulations
                (golfer_name, tournament_name, n_simulations, mean_earnings, median_earnings,
                 std_earnings, percentile_10, percentile_25, percentile_75, percentile_90,
                 win_count, top_10_count, cut_made_count, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.golfer_name,
                result.tournament_name,
                result.n_simulations,
                result.mean_earnings,
                result.median_earnings,
                result.std_earnings,
                result.percentile_10,
                result.percentile_25,
                result.percentile_75,
                result.percentile_90,
                result.win_count,
                result.top_10_count,
                result.cut_made_count,
                datetime.now().isoformat(),
            ))
            return cursor.lastrowid

    def get_simulation(self, golfer_name: str, tournament_name: str) -> Optional[SimulationResult]:
        """Get cached simulation result."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM simulations WHERE golfer_name = ? AND tournament_name = ?",
                (golfer_name, tournament_name)
            )
            row = cursor.fetchone()
            if row:
                return SimulationResult(
                    golfer_name=row["golfer_name"],
                    tournament_name=row["tournament_name"],
                    n_simulations=row["n_simulations"],
                    mean_earnings=row["mean_earnings"],
                    median_earnings=row["median_earnings"],
                    std_earnings=row["std_earnings"],
                    percentile_10=row["percentile_10"],
                    percentile_25=row["percentile_25"],
                    percentile_75=row["percentile_75"],
                    percentile_90=row["percentile_90"],
                    win_count=row["win_count"],
                    top_10_count=row["top_10_count"],
                    cut_made_count=row["cut_made_count"],
                )
        return None

    def clear_simulation_cache(self) -> int:
        """Clear all cached simulation results."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM simulations")
            return cursor.rowcount

    # =========================================================================
    # Golfer probabilities operations
    # =========================================================================

    def save_golfer_probability(
        self, golfer_name: str, tournament_name: str,
        win_prob: float = 0, top_5_prob: float = 0, top_10_prob: float = 0,
        top_20_prob: float = 0, make_cut_prob: float = 0
    ):
        """Save golfer probabilities for a tournament."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO golfer_probabilities
                (golfer_name, tournament_name, win_prob, top_5_prob, top_10_prob, top_20_prob, make_cut_prob, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                golfer_name, tournament_name, win_prob, top_5_prob,
                top_10_prob, top_20_prob, make_cut_prob, datetime.now().isoformat()
            ))

    def get_golfer_probability(self, golfer_name: str, tournament_name: str) -> Optional[Dict[str, float]]:
        """Get golfer probabilities for a tournament."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM golfer_probabilities WHERE golfer_name = ? AND tournament_name = ?",
                (golfer_name, tournament_name)
            )
            row = cursor.fetchone()
            if row:
                return {
                    "win_prob": row["win_prob"],
                    "top_5_prob": row["top_5_prob"],
                    "top_10_prob": row["top_10_prob"],
                    "top_20_prob": row["top_20_prob"],
                    "make_cut_prob": row["make_cut_prob"],
                }
        return None

    # =========================================================================
    # Available golfers operations
    # =========================================================================

    def save_available_golfers(self, golfer_names: List[str]):
        """Save list of available (unused) golfers."""
        with self._connection() as conn:
            cursor = conn.cursor()
            # Clear existing
            cursor.execute("DELETE FROM available_golfers")
            # Insert new
            now = datetime.now().isoformat()
            for name in golfer_names:
                cursor.execute(
                    "INSERT INTO available_golfers (golfer_name, updated_at) VALUES (?, ?)",
                    (name, now)
                )

    def get_available_golfers(self) -> List[str]:
        """Get list of available golfers."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT golfer_name FROM available_golfers")
            return [row["golfer_name"] for row in cursor.fetchall()]

    # =========================================================================
    # Cache operations
    # =========================================================================

    def set_cache(self, key: str, value: Any, expires_at: datetime):
        """Set a cache entry."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
                (key, json.dumps(value), expires_at.isoformat())
            )

    def get_cache(self, key: str) -> Optional[Any]:
        """Get a cache entry if not expired."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value, expires_at FROM cache WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                expires = datetime.fromisoformat(row["expires_at"])
                if expires > datetime.now():
                    try:
                        return json.loads(row["value"])
                    except json.JSONDecodeError:
                        # Invalid cache entry, delete it
                        cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                        return None
                # Expired, delete it
                cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
        return None

    def clear_expired_cache(self):
        """Remove all expired cache entries."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM cache WHERE expires_at < ?",
                (datetime.now().isoformat(),)
            )

    def clear_all_api_cache(self) -> int:
        """Clear all API-related cache entries."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM cache WHERE key LIKE 'datagolf:%'")
            return cursor.rowcount

    def get_golfer_count(self) -> int:
        """Get total number of golfers in database."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM golfers")
            return cursor.fetchone()[0]

    def get_valid_owgr_count(self) -> int:
        """Get count of golfers with valid OWGR (not 999)."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM golfers WHERE owgr < 999")
            return cursor.fetchone()[0]

    # =========================================================================
    # Utility operations
    # =========================================================================

    def get_total_earnings(self) -> int:
        """Get user's total earnings this season from standings."""
        with self._connection() as conn:
            cursor = conn.cursor()
            # Get from latest standings - look for user's team
            cursor.execute("""
                SELECT total_earnings FROM standings
                WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM standings)
                AND (username = 'Just a chip and a putt' OR player_name = 'Eric Gitberg')
                LIMIT 1
            """)
            result = cursor.fetchone()
            if result:
                return result[0]
            # Fallback to picks table sum
            cursor.execute("SELECT SUM(earnings) FROM picks")
            result = cursor.fetchone()[0]
            return result or 0

    def get_cuts_made_count(self) -> int:
        """Get user's total cuts made this season."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM picks WHERE made_cut = 1")
            return cursor.fetchone()[0]

    def get_picks_count(self) -> int:
        """Get number of picks made this season."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM picks")
            return cursor.fetchone()[0]

    # =========================================================================
    # Course History operations (Phase 2.1)
    # =========================================================================

    def save_course_history_entry(
        self,
        golfer_name: str,
        course_name: str,
        tournament_name: str,
        year: int,
        finish_position: int,
        earnings: int = 0,
        sg_total: float = 0.0,
        made_cut: bool = True
    ):
        """Save a single course history entry."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO course_history
                (golfer_name, course_name, tournament_name, year, finish_position, earnings, sg_total, made_cut, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                golfer_name, course_name, tournament_name, year,
                finish_position, earnings, sg_total,
                1 if made_cut else 0, datetime.now().isoformat()
            ))

    def get_course_history(self, golfer_name: str, course_name: str) -> Optional[CourseHistory]:
        """Get aggregated course history for a golfer at a course."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM course_history
                WHERE golfer_name = ? AND course_name = ?
                ORDER BY year DESC
            """, (golfer_name, course_name))
            rows = cursor.fetchall()

            if not rows:
                return None

            # Aggregate the data
            years_played = len(rows)
            finishes = [r["finish_position"] for r in rows if r["finish_position"]]
            earnings = sum(r["earnings"] for r in rows)
            sg_totals = [r["sg_total"] for r in rows if r["sg_total"] != 0]
            cuts_made = sum(1 for r in rows if r["made_cut"])
            missed_cuts = sum(1 for r in rows if not r["made_cut"])

            # Calculate stats
            avg_finish = sum(finishes) / len(finishes) if finishes else 0
            best_finish = min(finishes) if finishes else 999
            wins = sum(1 for f in finishes if f == 1)
            top_5s = sum(1 for f in finishes if f <= 5)
            top_10s = sum(1 for f in finishes if f <= 10)
            sg_avg = sum(sg_totals) / len(sg_totals) if sg_totals else 0

            # Recent performance (last 2 years)
            recent_rows = rows[:2]
            recent_finishes = [r["finish_position"] for r in recent_rows if r["finish_position"]]
            recent_sg_totals = [r["sg_total"] for r in recent_rows if r["sg_total"] != 0]
            recent_avg_finish = sum(recent_finishes) / len(recent_finishes) if recent_finishes else 0
            recent_sg = sum(recent_sg_totals) / len(recent_sg_totals) if recent_sg_totals else 0

            return CourseHistory(
                golfer_name=golfer_name,
                course_name=course_name,
                tournament_name=rows[0]["tournament_name"],
                years_played=years_played,
                avg_finish=avg_finish,
                best_finish=best_finish,
                wins=wins,
                top_5s=top_5s,
                top_10s=top_10s,
                cuts_made=cuts_made,
                missed_cuts=missed_cuts,
                total_earnings=earnings,
                sg_total_at_course=sg_avg,
                recent_avg_finish=recent_avg_finish,
                recent_sg=recent_sg,
            )

    def get_all_course_history_for_golfer(self, golfer_name: str) -> Dict[str, CourseHistory]:
        """Get course history for all courses a golfer has played."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT course_name FROM course_history
                WHERE golfer_name = ?
            """, (golfer_name,))
            courses = [row["course_name"] for row in cursor.fetchall()]

        result = {}
        for course in courses:
            history = self.get_course_history(golfer_name, course)
            if history:
                result[course] = history
        return result

    # =========================================================================
    # Golfer Availability operations (Phase 2.2)
    # =========================================================================

    def save_golfer_availability(
        self,
        golfer_name: str,
        tournament_name: str,
        status: str,
        probability: float = 0.0
    ):
        """Save golfer availability status for a tournament."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO golfer_availability
                (golfer_name, tournament_name, status, probability, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (golfer_name, tournament_name, status, probability, datetime.now().isoformat()))

    def get_golfer_availability(self, golfer_name: str, tournament_name: str) -> Optional[GolferAvailability]:
        """Get golfer availability status for a tournament."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT status FROM golfer_availability
                WHERE golfer_name = ? AND tournament_name = ?
            """, (golfer_name, tournament_name))
            row = cursor.fetchone()
            if row:
                try:
                    return GolferAvailability(row["status"])
                except ValueError:
                    return GolferAvailability.UNKNOWN
        return None

    def get_all_availability_for_tournament(self, tournament_name: str) -> Dict[str, GolferAvailability]:
        """Get availability status for all golfers in a tournament."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT golfer_name, status FROM golfer_availability
                WHERE tournament_name = ?
            """, (tournament_name,))
            result = {}
            for row in cursor.fetchall():
                try:
                    result[row["golfer_name"]] = GolferAvailability(row["status"])
                except ValueError:
                    result[row["golfer_name"]] = GolferAvailability.UNKNOWN
            return result

    # =========================================================================
    # Season Plan operations (Phase 2.3)
    # =========================================================================

    def save_season_plan_entry(
        self,
        tournament_name: str,
        tournament_date: date,
        golfer_name: Optional[str] = None,
        is_tentative: bool = True,
        projected_ev: float = 0.0,
        notes: str = "",
        entry_id: int = 1
    ):
        """Save or update a season plan entry."""
        with self._connection() as conn:
            cursor = conn.cursor()
            now = datetime.now().isoformat()
            cursor.execute("""
                INSERT OR REPLACE INTO season_plan
                (entry_id, tournament_name, tournament_date, golfer_name, is_tentative, projected_ev, notes, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry_id, tournament_name, tournament_date.isoformat(),
                golfer_name, 1 if is_tentative else 0, projected_ev, notes, now, now
            ))

    def get_season_plan(self, entry_id: int = 1) -> List[SeasonPlanEntry]:
        """Get the full season plan for an entry."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM season_plan
                WHERE entry_id = ?
                ORDER BY tournament_date
            """, (entry_id,))
            return [
                SeasonPlanEntry(
                    tournament_name=row["tournament_name"],
                    tournament_date=date.fromisoformat(row["tournament_date"]),
                    golfer_name=row["golfer_name"],
                    is_tentative=bool(row["is_tentative"]),
                    projected_ev=row["projected_ev"],
                    notes=row["notes"] or "",
                )
                for row in cursor.fetchall()
            ]

    def delete_season_plan_entry(self, tournament_name: str, entry_id: int = 1):
        """Delete a season plan entry."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM season_plan
                WHERE entry_id = ? AND tournament_name = ?
            """, (entry_id, tournament_name))

    def get_season_plan_conflicts(self, entry_id: int = 1) -> List[str]:
        """Find golfers assigned to multiple tournaments in the plan."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT golfer_name, COUNT(*) as count
                FROM season_plan
                WHERE entry_id = ? AND golfer_name IS NOT NULL
                GROUP BY golfer_name
                HAVING count > 1
            """, (entry_id,))
            return [row["golfer_name"] for row in cursor.fetchall()]

    def get_projected_season_earnings(self, entry_id: int = 1) -> float:
        """Calculate total projected earnings from season plan."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT SUM(projected_ev) as total
                FROM season_plan
                WHERE entry_id = ? AND golfer_name IS NOT NULL
            """, (entry_id,))
            result = cursor.fetchone()
            return result["total"] or 0.0

    # =========================================================================
    # Multi-Entry operations (Phase 3.2)
    # =========================================================================

    def create_entry(self, entry_name: str) -> int:
        """Create a new entry for multi-entry tracking."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO entries (entry_name, created_at)
                VALUES (?, ?)
            """, (entry_name, datetime.now().isoformat()))
            return cursor.lastrowid

    def get_all_entries(self) -> List[Entry]:
        """Get all entries."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM entries ORDER BY id")
            entries = []
            for row in cursor.fetchall():
                entry = Entry(
                    entry_id=row["id"],
                    entry_name=row["entry_name"],
                )
                # Get picks for this entry
                cursor.execute("""
                    SELECT golfer_name, tournament_name, tournament_date, earnings, position, made_cut
                    FROM entry_picks WHERE entry_id = ?
                    ORDER BY tournament_date
                """, (row["id"],))
                for pick_row in cursor.fetchall():
                    pick = Pick(
                        golfer_name=pick_row["golfer_name"],
                        tournament_name=pick_row["tournament_name"],
                        tournament_date=date.fromisoformat(pick_row["tournament_date"]),
                        earnings=pick_row["earnings"],
                        position=pick_row["position"],
                        made_cut=bool(pick_row["made_cut"]),
                    )
                    entry.picks.append(pick)
                    entry.total_earnings += pick.earnings
                    entry.used_golfers.append(pick.golfer_name)
                entries.append(entry)
            return entries

    def save_entry_pick(self, entry_id: int, pick: Pick):
        """Save a pick for a specific entry."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO entry_picks
                (entry_id, golfer_name, tournament_name, tournament_date, earnings, position, made_cut, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry_id, pick.golfer_name, pick.tournament_name,
                pick.tournament_date.isoformat(), pick.earnings,
                pick.position, 1 if pick.made_cut else 0, datetime.now().isoformat()
            ))

    def get_entry_used_golfers(self, entry_id: int) -> List[str]:
        """Get list of golfers used by a specific entry."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT golfer_name FROM entry_picks
                WHERE entry_id = ?
            """, (entry_id,))
            return [row["golfer_name"] for row in cursor.fetchall()]

    # =========================================================================
    # Learning Feature Operations
    # =========================================================================

    def save_pick_outcome(
        self,
        golfer_name: str,
        tournament_name: str,
        tournament_date: date,
        purse: int = 0,
        predicted_win_prob: float = 0,
        predicted_top10_prob: float = 0,
        predicted_ev: float = 0,
        predicted_course_fit: float = 0,
        strategic_score: float = 0,
        was_save_warning: bool = False,
        was_my_pick: bool = False
    ):
        """Record a prediction for later outcome comparison."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO pick_outcomes
                (golfer_name, tournament_name, tournament_date, purse,
                 predicted_win_prob, predicted_top10_prob, predicted_ev,
                 predicted_course_fit, strategic_score, was_save_warning,
                 was_my_pick, outcome_recorded, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """, (
                golfer_name, tournament_name, tournament_date.isoformat(),
                purse, predicted_win_prob, predicted_top10_prob, predicted_ev,
                predicted_course_fit, strategic_score,
                1 if was_save_warning else 0, 1 if was_my_pick else 0,
                datetime.now().isoformat()
            ))

    def record_pick_outcome_result(
        self,
        golfer_name: str,
        tournament_name: str,
        actual_position: int,
        actual_earnings: int,
        made_cut: bool
    ):
        """Record the actual outcome for a previously saved prediction."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE pick_outcomes
                SET actual_position = ?, actual_earnings = ?, made_cut = ?,
                    outcome_recorded = 1, outcome_recorded_at = ?
                WHERE golfer_name = ? AND tournament_name = ?
                  AND outcome_recorded = 0
            """, (
                actual_position, actual_earnings, 1 if made_cut else 0,
                datetime.now().isoformat(), golfer_name, tournament_name
            ))
            return cursor.rowcount > 0

    def get_pick_outcomes(self, days: int = 365, recorded_only: bool = False) -> List[Dict]:
        """Get pick outcomes for analysis."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cutoff = (datetime.now() - __import__('datetime').timedelta(days=days)).isoformat()
            sql = """
                SELECT * FROM pick_outcomes
                WHERE created_at >= ?
            """
            if recorded_only:
                sql += " AND outcome_recorded = 1"
            sql += " ORDER BY created_at DESC"
            cursor.execute(sql, (cutoff,))
            return [dict(row) for row in cursor.fetchall()]

    def get_pending_outcomes(self) -> List[Dict]:
        """Get predictions that need outcome recording."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM pick_outcomes
                WHERE outcome_recorded = 0
                ORDER BY tournament_date ASC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def save_learned_course_fit(
        self,
        tournament_name: str,
        skill_name: str,
        static_weight: float,
        learned_weight: float,
        confidence: float,
        sample_size: int
    ):
        """Save or update a learned course fit weight."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO learned_course_fits
                (tournament_name, skill_name, static_weight, learned_weight,
                 confidence, sample_size, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                tournament_name, skill_name, static_weight, learned_weight,
                confidence, sample_size, datetime.now().isoformat()
            ))

    def get_learned_course_fits(self, tournament_name: str = None) -> List[Dict]:
        """Get learned course fit weights."""
        with self._connection() as conn:
            cursor = conn.cursor()
            if tournament_name:
                cursor.execute("""
                    SELECT * FROM learned_course_fits
                    WHERE tournament_name = ?
                    ORDER BY skill_name
                """, (tournament_name,))
            else:
                cursor.execute("""
                    SELECT * FROM learned_course_fits
                    ORDER BY tournament_name, skill_name
                """)
            return [dict(row) for row in cursor.fetchall()]

    def save_learned_elite_tier(
        self,
        golfer_name: str,
        static_tier: int,
        learned_tier: int,
        performance_score: float,
        tier_confidence: float,
        wins_this_season: int = 0,
        top10s_this_season: int = 0,
        events_played: int = 0,
        total_earnings: float = 0
    ):
        """Save or update a learned elite tier adjustment."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO learned_elite_tiers
                (golfer_name, static_tier, learned_tier, performance_score,
                 tier_confidence, wins_this_season, top10s_this_season,
                 events_played, total_earnings, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                golfer_name, static_tier, learned_tier, performance_score,
                tier_confidence, wins_this_season, top10s_this_season,
                events_played, total_earnings, datetime.now().isoformat()
            ))

    def get_learned_elite_tiers(self) -> List[Dict]:
        """Get all learned elite tier adjustments."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM learned_elite_tiers
                ORDER BY learned_tier, performance_score DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_elite_tier_changes(self) -> List[Dict]:
        """Get golfers whose learned tier differs from static tier."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM learned_elite_tiers
                WHERE learned_tier != static_tier
                ORDER BY ABS(learned_tier - static_tier) DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def save_prediction(
        self,
        tournament_name: str,
        golfer_name: str,
        prediction_type: str,
        predicted_value: float
    ):
        """Save a prediction for accuracy tracking."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO predictions
                (tournament_name, golfer_name, prediction_type, predicted_value, prediction_made_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                tournament_name, golfer_name, prediction_type,
                predicted_value, datetime.now().isoformat()
            ))

    def record_prediction_outcome(
        self,
        tournament_name: str,
        golfer_name: str,
        prediction_type: str,
        actual_value: float
    ):
        """Record actual outcome for a prediction."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT predicted_value FROM predictions
                WHERE tournament_name = ? AND golfer_name = ? AND prediction_type = ?
            """, (tournament_name, golfer_name, prediction_type))
            row = cursor.fetchone()
            if row:
                predicted = row["predicted_value"]
                error = predicted - actual_value
                squared_error = error * error
                cursor.execute("""
                    UPDATE predictions
                    SET actual_value = ?, error = ?, squared_error = ?,
                        outcome_recorded_at = ?
                    WHERE tournament_name = ? AND golfer_name = ? AND prediction_type = ?
                """, (
                    actual_value, error, squared_error, datetime.now().isoformat(),
                    tournament_name, golfer_name, prediction_type
                ))

    def get_predictions(self, prediction_type: str = None, days: int = 365) -> List[Dict]:
        """Get predictions for analysis."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cutoff = (datetime.now() - __import__('datetime').timedelta(days=days)).isoformat()
            if prediction_type:
                cursor.execute("""
                    SELECT * FROM predictions
                    WHERE prediction_type = ? AND prediction_made_at >= ?
                    ORDER BY prediction_made_at DESC
                """, (prediction_type, cutoff))
            else:
                cursor.execute("""
                    SELECT * FROM predictions
                    WHERE prediction_made_at >= ?
                    ORDER BY prediction_made_at DESC
                """, (cutoff,))
            return [dict(row) for row in cursor.fetchall()]

    def save_model_accuracy(
        self,
        metric_name: str,
        metric_value: float,
        tournament_name: str = None,
        sample_size: int = None,
        time_period: str = "weekly"
    ):
        """Save a model accuracy metric."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO model_accuracy
                (metric_name, metric_value, tournament_name, sample_size, time_period, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                metric_name, metric_value, tournament_name, sample_size,
                time_period, datetime.now().isoformat()
            ))

    def get_model_accuracy_history(self, metric_name: str = None, days: int = 180) -> List[Dict]:
        """Get model accuracy history for trend analysis."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cutoff = (datetime.now() - __import__('datetime').timedelta(days=days)).isoformat()
            if metric_name:
                cursor.execute("""
                    SELECT * FROM model_accuracy
                    WHERE metric_name = ? AND recorded_at >= ?
                    ORDER BY recorded_at ASC
                """, (metric_name, cutoff))
            else:
                cursor.execute("""
                    SELECT * FROM model_accuracy
                    WHERE recorded_at >= ?
                    ORDER BY recorded_at ASC
                """, (cutoff,))
            return [dict(row) for row in cursor.fetchall()]

    def save_opponent_pattern(
        self,
        opponent_name: str,
        prefers_favorites: float = 0.5,
        prefers_value: float = 0.5,
        prefers_course_fit: float = 0.5,
        risk_tolerance: float = 0.5,
        avg_golfer_ranking: float = None,
        avg_win_prob_selected: float = None,
        total_picks_tracked: int = 0
    ):
        """Save or update an opponent's picking pattern."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO opponent_patterns
                (opponent_name, prefers_favorites, prefers_value, prefers_course_fit,
                 risk_tolerance, avg_golfer_ranking, avg_win_prob_selected,
                 total_picks_tracked, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                opponent_name, prefers_favorites, prefers_value, prefers_course_fit,
                risk_tolerance, avg_golfer_ranking, avg_win_prob_selected,
                total_picks_tracked, datetime.now().isoformat()
            ))

    def get_opponent_patterns(self) -> List[Dict]:
        """Get all learned opponent patterns."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM opponent_patterns
                ORDER BY total_picks_tracked DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def get_opponent_pattern(self, opponent_name: str) -> Optional[Dict]:
        """Get a specific opponent's pattern."""
        with self._connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM opponent_patterns WHERE opponent_name = ?
            """, (opponent_name,))
            row = cursor.fetchone()
            return dict(row) if row else None
