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

from .models import (
    Tournament, Golfer, GolferStats, Pick, LeagueStanding,
    OpponentPick, SimulationResult, Tier
)
from .config import get_config


class Database:
    """SQLite database manager."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database connection."""
        config = get_config()
        self.db_path = db_path or config.db_path
        self._init_db()

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
        stats_data = json.loads(row["stats_json"]) if row["stats_json"] else {}
        course_history = json.loads(row["course_history_json"]) if row["course_history_json"] else {}
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
            cursor.execute("""
                INSERT OR REPLACE INTO opponent_picks
                (opponent_username, golfer_name, tournament_name, tournament_date, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                pick.opponent_username,
                pick.golfer_name,
                pick.tournament_name,
                pick.tournament_date.isoformat(),
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
                    return json.loads(row["value"])
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

    # =========================================================================
    # Utility operations
    # =========================================================================

    def get_total_earnings(self) -> int:
        """Get user's total earnings this season."""
        with self._connection() as conn:
            cursor = conn.cursor()
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
