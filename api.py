"""
Data Golf API client for PGA One and Done Optimizer.
Fetches golfer predictions, odds, and statistics.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import requests

try:
    from .config import get_config
    from .database import Database
    from .models import Golfer, GolferStats, ApproachBuckets, GolferAvailability
except ImportError:
    from config import get_config
    from database import Database
    from models import Golfer, GolferStats, ApproachBuckets, GolferAvailability

logger = logging.getLogger(__name__)


@dataclass
class PredictionData:
    """Prediction data for a golfer in a tournament."""
    golfer_name: str
    datagolf_id: str
    win_prob: float
    top_5_prob: float
    top_10_prob: float
    top_20_prob: float
    make_cut_prob: float
    expected_position: float


class DataGolfAPI:
    """Client for Data Golf API."""

    BASE_URL = "https://feeds.datagolf.com"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize API client."""
        config = get_config()
        self.api_key = api_key or config.datagolf_api_key
        self.db = Database()
        self._session = requests.Session()
        self.last_error: str = ""  # Track last error for UI display

    def _request(self, endpoint: str, params: Optional[Dict] = None, cache_hours: int = 1) -> Optional[Dict]:
        """Make API request with caching."""
        if not self.api_key:
            self.last_error = "DATAGOLF_API_KEY not configured"
            raise ValueError(
                "DATAGOLF_API_KEY not configured. "
                "Set the DATAGOLF_API_KEY environment variable. "
                "Get a key at https://datagolf.com/api-access"
            )

        # Check cache first
        cache_key = f"datagolf:{endpoint}:{str(params)}"
        cached = self.db.get_cache(cache_key)
        if cached:
            logger.debug(f"Using cached data for {endpoint}")
            return cached

        # Make request with retry logic
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params["key"] = self.api_key

        max_retries = 3
        base_delay = 1.0  # seconds

        for attempt in range(max_retries):
            try:
                response = self._session.get(url, params=params, timeout=30)
                response.raise_for_status()

                # Parse JSON with error handling
                try:
                    data = response.json()
                except json.JSONDecodeError as e:
                    self.last_error = f"Failed to parse JSON from {endpoint}: {e}"
                    logger.error(self.last_error)
                    return None

                # Only cache non-empty responses
                if data and (isinstance(data, dict) and len(data) > 0 or isinstance(data, list) and len(data) > 0):
                    expires = datetime.now() + timedelta(hours=cache_hours)
                    self.db.set_cache(cache_key, data, expires)
                    self.last_error = ""  # Clear error on success
                else:
                    self.last_error = f"Empty response from {endpoint}"
                    logger.warning(self.last_error)

                return data
            except requests.RequestException as e:
                self.last_error = f"API request failed: {e}"
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"API request failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"API request failed after {max_retries} attempts: {e}")
                    return None

        return None

    def get_pre_tournament_predictions(self, tour: str = "pga") -> List[PredictionData]:
        """
        Get pre-tournament predictions for upcoming event.
        Returns win/top-5/top-10/top-20/make-cut probabilities.
        """
        data = self._request(
            "/preds/pre-tournament",
            params={"tour": tour, "file_format": "json"},
            cache_hours=1
        )

        if not data:
            return []

        predictions = []
        # Try both keys - API may return either
        baseline_preds = data.get("baseline_history_fit") or data.get("baseline", [])

        for player in baseline_preds:
            # API uses 'win', 'top_5', etc. (without _prob suffix)
            pred = PredictionData(
                golfer_name=player.get("player_name", ""),
                datagolf_id=str(player.get("dg_id", "")),
                win_prob=player.get("win_prob") or player.get("win", 0),
                top_5_prob=player.get("top_5_prob") or player.get("top_5", 0),
                top_10_prob=player.get("top_10_prob") or player.get("top_10", 0),
                top_20_prob=player.get("top_20_prob") or player.get("top_20", 0),
                make_cut_prob=player.get("make_cut_prob") or player.get("make_cut", 0),
                expected_position=player.get("expected_place", 50),
            )
            predictions.append(pred)

            # Save to database
            self.db.save_golfer_probability(
                golfer_name=pred.golfer_name,
                tournament_name=data.get("event_name", "Unknown"),
                win_prob=pred.win_prob,
                top_5_prob=pred.top_5_prob,
                top_10_prob=pred.top_10_prob,
                top_20_prob=pred.top_20_prob,
                make_cut_prob=pred.make_cut_prob,
            )

        logger.info(f"Fetched predictions for {len(predictions)} golfers")
        return predictions

    def get_skill_ratings(self, tour: str = "pga") -> Dict[str, Dict[str, float]]:
        """
        Get current skill ratings (strokes gained components).
        Returns dict of golfer_name -> skill metrics.
        """
        data = self._request(
            "/preds/skill-ratings",
            params={"tour": tour, "file_format": "json"},
            cache_hours=24
        )

        if not data:
            return {}

        ratings = {}
        for player in data.get("players", []):
            name = player.get("player_name", "")
            if name:
                ratings[name] = {
                    "sg_total": player.get("sg_total", 0),
                    "sg_off_tee": player.get("sg_ott", 0),
                    "sg_approach": player.get("sg_app", 0),
                    "sg_around_green": player.get("sg_arg", 0),
                    "sg_putting": player.get("sg_putt", 0),
                    "driving_distance": player.get("driving_distance", 0),
                    "driving_accuracy": player.get("driving_acc", 0),
                }
        return ratings

    def get_player_list(self) -> List[Dict[str, Any]]:
        """Get full player list with IDs and rankings."""
        data = self._request(
            "/get-player-list",
            params={"file_format": "json"},
            cache_hours=24
        )

        if not data:
            return []

        players = []
        for player in data:
            players.append({
                "name": player.get("player_name", ""),
                "datagolf_id": str(player.get("dg_id", "")),
                "owgr": player.get("owgr", 999),
                "country": player.get("country", ""),
            })
        return players

    def get_dg_rankings(self) -> Dict[str, Dict[str, Any]]:
        """
        Get Data Golf rankings with proper OWGR data.
        Returns top 500 players with skill estimates and OWGR rank.
        """
        data = self._request(
            "/preds/get-dg-rankings",
            params={"file_format": "json"},
            cache_hours=12
        )

        if not data:
            return {}

        rankings = {}
        rankings_list = data.get("rankings", data if isinstance(data, list) else [])
        for player in rankings_list:
            name = player.get("player_name", "")
            if name:
                rankings[name] = {
                    "dg_rank": player.get("datagolf_rank", 999),
                    "owgr": player.get("owgr_rank", 999),
                    "dg_skill": player.get("dg_skill_estimate", 0),
                    "datagolf_id": str(player.get("dg_id", "")),
                }
        return rankings

    def get_field_updates(self, tour: str = "pga") -> List[str]:
        """Get current field for upcoming tournament."""
        data = self._request(
            "/field-updates",
            params={"tour": tour, "file_format": "json"},
            cache_hours=1
        )

        if not data:
            return []

        field = []
        for player in data.get("field", []):
            name = player.get("player_name", "")
            if name:
                field.append(name)
        return field

    def get_historical_odds(self, tour: str = "pga", market: str = "win") -> Dict[str, float]:
        """
        Get historical betting odds.
        market: 'win', 'top_5', 'top_10', 'top_20', 'make_cut'
        """
        data = self._request(
            "/historical-odds/outrights",
            params={"tour": tour, "market": market, "file_format": "json"},
            cache_hours=6
        )

        if not data:
            return {}

        odds = {}
        for player in data.get("odds", []):
            name = player.get("player_name", "")
            prob = player.get("implied_prob", 0)
            if name:
                odds[name] = prob
        return odds

    def get_live_tournament_stats(self, tour: str = "pga") -> Dict[str, Any]:
        """Get live tournament statistics (if tournament in progress)."""
        data = self._request(
            "/preds/in-play",
            params={"tour": tour, "file_format": "json"},
            cache_hours=0.25  # 15 minute cache for live data
        )
        return data or {}

    def get_approach_skill(self, tour: str = "pga") -> Dict[str, ApproachBuckets]:
        """
        Get approach shot skill by yardage bucket.
        Returns strokes gained per yardage range (50-100, 100-150, 150-200, 200+).
        """
        data = self._request(
            "/preds/approach-skill",
            params={"file_format": "json"},
            cache_hours=24
        )

        if not data:
            return {}

        approach_data = {}
        # API uses 'data' key with field names like '50_100_fw_sg_per_shot'
        players_list = data.get("data", [])
        for player in players_list:
            name = player.get("player_name", "")
            if name:
                approach_data[name] = ApproachBuckets(
                    sg_50_100=player.get("50_100_fw_sg_per_shot", 0),
                    sg_100_150=player.get("100_150_fw_sg_per_shot", 0),
                    sg_150_200=player.get("150_200_fw_sg_per_shot", 0),
                    sg_200_plus=player.get("over_200_fw_sg_per_shot", 0),
                    sg_fairway=player.get("over_150_rgh_sg_per_shot", 0),  # Rough over 150
                    sg_rough=player.get("under_150_rgh_sg_per_shot", 0),   # Rough under 150
                )
        return approach_data

    def get_betting_outrights(self, tour: str = "pga", market: str = "win") -> Dict[str, Dict[str, float]]:
        """
        Get betting odds from multiple sportsbooks.
        market: 'win', 'top_5', 'top_10', 'top_20', 'make_cut'
        Returns dict of golfer_name -> {book: implied_prob, consensus: avg_prob}
        """
        data = self._request(
            "/betting-tools/outrights",
            params={"tour": tour, "market": market, "file_format": "json"},
            cache_hours=1
        )

        if not data:
            return {}

        odds = {}
        for player in data.get("odds", []):
            name = player.get("player_name", "")
            if name:
                books = {}
                for book, prob in player.get("books", {}).items():
                    books[book] = prob
                # Calculate consensus (average of all books)
                book_probs = list(books.values())
                consensus = sum(book_probs) / len(book_probs) if book_probs else 0
                books["consensus"] = consensus
                # Data Golf model probability
                books["datagolf_model"] = player.get("dg_prob", 0)
                odds[name] = books
        return odds

    def get_betting_matchups(self, tour: str = "pga") -> List[Dict[str, Any]]:
        """
        Get head-to-head match-up and 3-ball betting odds.
        Useful for identifying relative value.
        """
        data = self._request(
            "/betting-tools/matchups",
            params={"tour": tour, "market": "tournament_matchups", "file_format": "json"},
            cache_hours=1
        )

        if not data:
            return []

        matchups = []
        for m in data.get("matchups", []):
            matchups.append({
                "type": m.get("type", "2ball"),  # 2ball or 3ball
                "player1": m.get("player1_name", ""),
                "player1_prob": m.get("player1_win_prob", 0),
                "player2": m.get("player2_name", ""),
                "player2_prob": m.get("player2_win_prob", 0),
                "player3": m.get("player3_name", ""),  # For 3-ball
                "player3_prob": m.get("player3_win_prob", 0),
                "tie_prob": m.get("tie_prob", 0),
            })
        return matchups

    def get_fantasy_projections(self, site: str = "draftkings") -> Dict[str, Dict[str, float]]:
        """
        Get fantasy golf projections for DraftKings/FanDuel.
        site: 'draftkings' or 'fanduel'
        """
        data = self._request(
            "/fantasy-projection",
            params={"site": site, "file_format": "json"},
            cache_hours=2
        )

        if not data:
            return {}

        projections = {}
        for player in data.get("projections", []):
            name = player.get("player_name", "")
            if name:
                projections[name] = {
                    "salary": player.get("salary", 0),
                    "projected_points": player.get("projected_points", 0),
                    "ownership_pct": player.get("ownership_pct", 0),
                    "value": player.get("value", 0),  # points per $1000
                    "ceiling": player.get("ceiling", 0),
                    "floor": player.get("floor", 0),
                }
        return projections

    def get_course_fit_predictions(self, tour: str = "pga", tournament_name: str = None) -> Dict[str, float]:
        """
        Calculate course fit adjustments based on golfer skills matching course characteristics.
        Returns strokes gained adjustment per round based on course fit.

        Uses local course profiles and golfer stats rather than Data Golf comparison
        which often returns neutral fits.
        """
        try:
            from config import get_course_profile, get_next_tournament, get_tournament_by_name
        except ImportError:
            from .config import get_course_profile, get_next_tournament, get_tournament_by_name

        # Get the tournament to determine course
        if tournament_name:
            tournament = get_tournament_by_name(tournament_name)
        else:
            tournament = get_next_tournament()

        if not tournament:
            return {}

        profile = get_course_profile(tournament.course)
        if not profile:
            return {}

        # Get all golfers from database
        golfers = self.db.get_all_golfers()
        if not golfers:
            return {}

        fit_adjustments = {}

        for golfer in golfers:
            stats = golfer.stats
            if not stats:
                continue

            # Calculate fit score based on skill-course matching
            total_adjustment = 0.0

            # === DRIVING ===
            # Tour average driving distance ~295 yards
            if stats.driving_distance > 0:
                dist_above_avg = (stats.driving_distance - 295) / 10
                distance_fit = dist_above_avg * 0.15 * profile.driving_distance
                total_adjustment += distance_fit

            # Driving accuracy - tour average ~60%
            if stats.driving_accuracy > 0:
                acc_above_avg = (stats.driving_accuracy - 60) / 10
                accuracy_fit = acc_above_avg * 0.12 * profile.driving_accuracy
                total_adjustment += accuracy_fit

            # === APPROACH ===
            if stats.sg_approach != 0:
                approach_weight = (profile.approach_long + profile.approach_mid + profile.approach_short) / 3
                approach_fit = stats.sg_approach * 0.5 * approach_weight
                total_adjustment += approach_fit

            # Approach by yardage buckets
            if stats.approach_buckets:
                buckets = stats.approach_buckets
                if buckets.sg_200_plus != 0 and profile.approach_long > 0.3:
                    total_adjustment += buckets.sg_200_plus * 0.3 * profile.approach_long
                if buckets.sg_100_150 != 0 and profile.approach_short > 0.3:
                    total_adjustment += buckets.sg_100_150 * 0.25 * profile.approach_short

            # === SHORT GAME ===
            if stats.sg_around_green != 0:
                total_adjustment += stats.sg_around_green * 0.4 * profile.around_green

            # === PUTTING ===
            if stats.sg_putting != 0:
                total_adjustment += stats.sg_putting * 0.35 * profile.putting

            # === WIND FACTOR ===
            if profile.wind_factor > 0.5 and stats.driving_accuracy > 0:
                wind_bonus = ((stats.driving_accuracy - 55) / 15) * 0.1 * profile.wind_factor
                total_adjustment += wind_bonus

            # === ROUGH PENALTY ===
            if profile.rough_penalty > 0.5 and stats.driving_accuracy > 0:
                rough_impact = ((stats.driving_accuracy - 60) / 20) * 0.15 * profile.rough_penalty
                total_adjustment += rough_impact

            # Cap adjustment to reasonable range
            total_adjustment = max(-0.8, min(0.8, total_adjustment))

            if total_adjustment != 0:
                fit_adjustments[golfer.name] = total_adjustment

        return fit_adjustments

    def get_live_win_probabilities(self, tour: str = "pga") -> List[Dict[str, Any]]:
        """
        Get live/in-play win probabilities during tournament.
        Returns current leaderboard with updated win probabilities.
        """
        data = self._request(
            "/preds/in-play",
            params={"tour": tour, "file_format": "json"},
            cache_hours=0.1  # 6 minute cache for live data
        )

        if not data:
            return []

        players = []
        for player in data.get("leaderboard", []):
            players.append({
                "name": player.get("player_name", ""),
                "position": player.get("position", 0),
                "score": player.get("total", 0),
                "thru": player.get("thru", 0),
                "round_score": player.get("round", 0),
                "win_prob": player.get("win_prob", 0),
                "top_5_prob": player.get("top_5_prob", 0),
                "top_10_prob": player.get("top_10_prob", 0),
                "make_cut_prob": player.get("make_cut_prob", 1.0),
            })
        return players

    def sync_golfers_to_db(self) -> int:
        """
        Sync golfer data from API to database.
        Returns number of golfers synced.
        """
        # Clear stale simulation cache when syncing new data
        cleared = self.db.clear_simulation_cache()
        if cleared > 0:
            logger.info(f"Cleared {cleared} stale simulation cache entries")

        # Get player list
        players = self.get_player_list()
        if not players:
            logger.warning("No players returned from API")
            return 0

        # Get Data Golf rankings (has proper OWGR data)
        dg_rankings = self.get_dg_rankings()

        # Get skill ratings
        skills = self.get_skill_ratings()

        # Get approach skill data
        approach_skills = self.get_approach_skill()

        # Get course fit adjustments
        course_fits = self.get_course_fit_predictions()

        # Get predictions as fallback for OWGR
        predictions = self.get_pre_tournament_predictions()
        pred_ranking = {}
        sorted_preds = sorted(predictions, key=lambda p: p.win_prob, reverse=True)
        for rank, pred in enumerate(sorted_preds, 1):
            pred_ranking[pred.golfer_name] = rank

        count = 0
        for player in players:
            name = player["name"]
            skill_data = skills.get(name, {})
            approach_data = approach_skills.get(name)
            course_fit = course_fits.get(name, 0)

            # Priority for OWGR: 1) DG Rankings, 2) Player List, 3) Prediction ranking
            dg_rank_data = dg_rankings.get(name, {})
            owgr = dg_rank_data.get("owgr", 999)
            if owgr == 999 or owgr is None:
                owgr = player.get("owgr", 999)
            if owgr == 999 or owgr is None:
                owgr = pred_ranking.get(name, 999)

            golfer = Golfer(
                name=name,
                owgr=owgr,
                datagolf_id=player.get("datagolf_id"),
                stats=GolferStats(
                    sg_total=skill_data.get("sg_total", 0),
                    sg_off_tee=skill_data.get("sg_off_tee", 0),
                    sg_approach=skill_data.get("sg_approach", 0),
                    sg_around_green=skill_data.get("sg_around_green", 0),
                    sg_putting=skill_data.get("sg_putting", 0),
                    driving_distance=skill_data.get("driving_distance", 0),
                    driving_accuracy=skill_data.get("driving_accuracy", 0),
                    approach_buckets=approach_data,
                ),
                course_fit_adjustment=course_fit,
            )
            self.db.save_golfer(golfer)
            count += 1

        logger.info(f"Synced {count} golfers to database")
        return count

    def get_golfer_with_predictions(self, name: str, tournament_name: str = "") -> Optional[Golfer]:
        """
        Get a golfer with current predictions loaded.
        """
        golfer = self.db.get_golfer(name)
        if not golfer:
            return None

        # Try to get probabilities
        probs = self.db.get_golfer_probability(name, tournament_name)
        if probs:
            golfer.win_probability = probs.get("win_prob", 0)
            golfer.top_10_probability = probs.get("top_10_prob", 0)
            golfer.top_20_probability = probs.get("top_20_prob", 0)
            golfer.make_cut_probability = probs.get("make_cut_prob", 0)

        return golfer

    def get_tournament_field_with_predictions(self, tournament_name: str = "") -> List[Golfer]:
        """
        Get full field with predictions for upcoming tournament.
        """
        # Fetch fresh predictions
        predictions = self.get_pre_tournament_predictions()

        golfers = []
        for pred in predictions:
            golfer = self.db.get_golfer(pred.golfer_name)
            if not golfer:
                golfer = Golfer(
                    name=pred.golfer_name,
                    datagolf_id=pred.datagolf_id,
                )

            golfer.win_probability = pred.win_prob
            golfer.top_10_probability = pred.top_10_prob
            golfer.top_20_probability = pred.top_20_prob
            golfer.make_cut_probability = pred.make_cut_prob

            golfers.append(golfer)

        return sorted(golfers, key=lambda g: g.win_probability, reverse=True)

    def health_check(self) -> bool:
        """Check if API is accessible and key is valid."""
        if not self.api_key:
            return False

        try:
            response = self._session.get(
                f"{self.BASE_URL}/get-player-list",
                params={"key": self.api_key, "file_format": "json"},
                timeout=10
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

    # =========================================================================
    # Player Decompositions - Course Fit from Data Golf
    # =========================================================================

    def get_player_decompositions(self, tour: str = "pga") -> Dict[str, Dict[str, float]]:
        """
        Get detailed player skill decompositions with REAL course fit adjustments.
        This is Data Golf's actual course fit prediction, not our calculated estimate.

        Returns dict of golfer_name -> {
            baseline_pred: base skill prediction,
            course_history_adj: adjustment from past performance at this course,
            course_fit_adj: adjustment based on skill-course matching,
            total_pred: final prediction (baseline + adjustments)
        }
        """
        data = self._request(
            "/preds/player-decompositions",
            params={"tour": tour, "file_format": "json"},
            cache_hours=2
        )

        if not data:
            return {}

        decomps = {}
        players = data.get("players", [])
        for player in players:
            name = player.get("player_name", "")
            if name:
                decomps[name] = {
                    "baseline_pred": player.get("baseline_pred", 0),
                    "course_history_adj": player.get("course_history_adj", 0),
                    "course_fit_adj": player.get("course_fit_adj", 0),
                    "total_pred": player.get("total_pred", 0),
                    "sg_baseline": player.get("sg_baseline", 0),
                    "sg_course_history": player.get("sg_course_history", 0),
                    "sg_course_fit": player.get("sg_course_fit", 0),
                }

        logger.info(f"Fetched decompositions for {len(decomps)} players")
        return decomps

    def get_live_strokes_gained(self, tour: str = "pga") -> Dict[str, Dict[str, float]]:
        """
        Get live strokes-gained stats during tournament play.

        Returns dict of golfer_name -> {
            sg_putt, sg_arg, sg_app, sg_ott, sg_t2g, sg_total,
            distance, accuracy, gir, prox_fw, prox_rgh, scrambling
        }
        """
        data = self._request(
            "/preds/live-tournament-stats",
            params={"tour": tour, "file_format": "json", "stats": "sg"},
            cache_hours=0.1  # 6 minute cache for live data
        )

        if not data:
            return {}

        stats = {}
        live_stats = data.get("live_stats", [])
        for player in live_stats:
            name = player.get("player_name", "")
            if name:
                stats[name] = {
                    "sg_putt": player.get("sg_putt", 0),
                    "sg_arg": player.get("sg_arg", 0),
                    "sg_app": player.get("sg_app", 0),
                    "sg_ott": player.get("sg_ott", 0),
                    "sg_t2g": player.get("sg_t2g", 0),
                    "sg_total": player.get("sg_total", 0),
                    "driving_distance": player.get("distance", 0),
                    "driving_accuracy": player.get("accuracy", 0),
                    "gir": player.get("gir", 0),
                    "scrambling": player.get("scrambling", 0),
                }
        return stats

    # =========================================================================
    # Phase 2.1: Historical Course Performance
    # =========================================================================

    def get_historical_results(
        self,
        tournament_name: str = None,
        years: int = 5
    ) -> Dict[str, List[Dict]]:
        """
        Get historical tournament results for the last N years.
        Returns dict of golfer_name -> list of {year, position, earnings, sg_total}.

        Note: Data Golf doesn't have a direct historical results endpoint,
        so we'll use the historical raw data endpoint for tournament archives.
        """
        data = self._request(
            "/historical-raw-data/event-list",
            params={"file_format": "json"},
            cache_hours=168  # Cache for 1 week
        )

        if not data:
            return {}

        # Filter events matching the tournament name
        matching_events = []
        events = data if isinstance(data, list) else data.get("events", [])
        for event in events:
            event_name = event.get("event_name", "")
            if tournament_name and tournament_name.lower() in event_name.lower():
                matching_events.append(event)

        # Get results for each matching event
        results_by_golfer = {}
        for event in matching_events[-years:]:  # Last N years
            event_id = event.get("event_id")
            if not event_id:
                continue

            event_data = self._request(
                "/historical-raw-data/rounds",
                params={"event_id": event_id, "file_format": "json"},
                cache_hours=168
            )

            if not event_data:
                continue

            rounds = event_data.get("rounds", event_data if isinstance(event_data, list) else [])
            # Aggregate to tournament level
            golfer_totals = {}
            for round_data in rounds:
                name = round_data.get("player_name", "")
                if not name:
                    continue
                if name not in golfer_totals:
                    golfer_totals[name] = {
                        "year": event.get("calendar_year", 0),
                        "position": round_data.get("fin_position", 999),
                        "earnings": round_data.get("earnings", 0),
                        "sg_total": round_data.get("sg_total", 0),
                        "made_cut": round_data.get("made_cut", True),
                    }
                else:
                    # Update with final position/earnings
                    if round_data.get("fin_position"):
                        golfer_totals[name]["position"] = round_data.get("fin_position")
                    if round_data.get("earnings"):
                        golfer_totals[name]["earnings"] = round_data.get("earnings")

            for golfer_name, totals in golfer_totals.items():
                if golfer_name not in results_by_golfer:
                    results_by_golfer[golfer_name] = []
                results_by_golfer[golfer_name].append(totals)

        return results_by_golfer

    def sync_course_history_to_db(self, tournament_name: str, course_name: str, years: int = 5) -> int:
        """
        Fetch historical results and save to database.
        Returns number of records saved.
        """
        historical = self.get_historical_results(tournament_name, years)
        count = 0

        for golfer_name, results in historical.items():
            for result in results:
                self.db.save_course_history_entry(
                    golfer_name=golfer_name,
                    course_name=course_name,
                    tournament_name=tournament_name,
                    year=result.get("year", 0),
                    finish_position=result.get("position", 999),
                    earnings=result.get("earnings", 0),
                    sg_total=result.get("sg_total", 0),
                    made_cut=result.get("made_cut", True),
                )
                count += 1

        logger.info(f"Synced {count} course history records for {tournament_name}")
        return count

    # =========================================================================
    # Phase 2.2: Golfer Availability Check
    # =========================================================================

    def get_field_predictions(self, tour: str = "pga") -> Dict[str, Dict]:
        """
        Get pre-tournament field predictions with commitment likelihood.
        Returns dict of golfer_name -> {status, probability}.
        """
        data = self._request(
            "/field-updates",
            params={"tour": tour, "file_format": "json"},
            cache_hours=1
        )

        if not data:
            return {}

        result = {}
        for player in data.get("field", []):
            name = player.get("player_name", "")
            if not name:
                continue

            # Data Golf field updates include commitment status
            status = player.get("status", "unknown")
            prob = player.get("in_prob", 0.5)  # Probability of being in field

            # Map to our availability enum
            if status == "in" or prob >= 0.95:
                availability = GolferAvailability.CONFIRMED
            elif prob >= 0.75:
                availability = GolferAvailability.LIKELY
            elif prob >= 0.50:
                availability = GolferAvailability.UNLIKELY
            elif status == "out" or prob < 0.25:
                availability = GolferAvailability.OUT
            else:
                availability = GolferAvailability.UNKNOWN

            result[name] = {
                "status": availability,
                "probability": prob,
            }

        return result

    def sync_availability_to_db(self, tournament_name: str) -> int:
        """
        Sync field predictions/availability to database.
        Returns number of records saved.
        """
        predictions = self.get_field_predictions()
        count = 0

        for golfer_name, pred in predictions.items():
            self.db.save_golfer_availability(
                golfer_name=golfer_name,
                tournament_name=tournament_name,
                status=pred["status"].value,
                probability=pred["probability"],
            )
            count += 1

        logger.info(f"Synced {count} availability records for {tournament_name}")
        return count

    # =========================================================================
    # NEW ENDPOINTS - Tour Schedule & Archive
    # =========================================================================

    def get_tour_schedule(
        self,
        tour: str = "pga",
        season: int = 2026,
        upcoming_only: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get tour schedule with event names, courses, locations, winners.

        Args:
            tour: pga, euro, kft, alt
            season: 2024-2026
            upcoming_only: Only return upcoming events
        """
        data = self._request(
            "/get-schedule",
            params={
                "tour": tour,
                "season": str(season),
                "upcoming_only": "yes" if upcoming_only else "no",
                "file_format": "json"
            },
            cache_hours=24
        )

        if not data:
            return []

        return data.get("schedule", [])

    def get_pre_tournament_archive(
        self,
        event_id: str,
        year: int = 2025
    ) -> Dict[str, Any]:
        """
        Get historical pre-tournament predictions for a past event.

        Args:
            event_id: Event ID from schedule
            year: 2020-2025
        """
        data = self._request(
            "/preds/pre-tournament-archive",
            params={
                "event_id": event_id,
                "year": str(year),
                "odds_format": "percent",
                "file_format": "json"
            },
            cache_hours=168  # Cache for a week (historical data)
        )

        return data or {}

    # =========================================================================
    # NEW ENDPOINTS - Live Tournament Data
    # =========================================================================

    def get_live_in_play(self, tour: str = "pga") -> Dict[str, Any]:
        """
        Get live finish probabilities during tournaments (updates every 5 min).

        Returns current standings with live win/top-10/cut probabilities.
        """
        data = self._request(
            "/preds/in-play",
            params={
                "tour": tour,
                "dead_heat": "no",
                "odds_format": "percent",
                "file_format": "json"
            },
            cache_hours=0  # Don't cache live data
        )

        return data or {}

    def get_live_hole_stats(self, tour: str = "pga") -> Dict[str, Any]:
        """
        Get live hole scoring averages and distributions by tee time wave.

        Returns birdie/par/bogey rates per hole, useful for weather impact analysis.
        """
        data = self._request(
            "/preds/live-hole-stats",
            params={
                "tour": tour,
                "file_format": "json"
            },
            cache_hours=0  # Don't cache live data
        )

        return data or {}

    # =========================================================================
    # NEW ENDPOINTS - Betting Tools (Extended)
    # =========================================================================

    def get_matchups_all_pairings(self, tour: str = "pga") -> List[Dict[str, Any]]:
        """
        Get Data Golf matchup/3-ball odds for every possible pairing in next round.

        Useful for finding betting value across all matchup combinations.
        """
        data = self._request(
            "/betting-tools/matchups-all-pairings",
            params={
                "tour": tour,
                "odds_format": "percent",
                "file_format": "json"
            },
            cache_hours=1
        )

        if not data:
            return []

        return data.get("matchups", [])

    # =========================================================================
    # NEW ENDPOINTS - Historical Raw Data
    # =========================================================================

    def get_historical_event_list(self, tour: str = "pga") -> List[Dict[str, Any]]:
        """
        Get list of historical events with IDs for use with rounds endpoint.

        Args:
            tour: pga, euro, kft, liv, and 20+ other tour codes
        """
        data = self._request(
            "/historical-raw-data/event-list",
            params={
                "tour": tour,
                "file_format": "json"
            },
            cache_hours=168  # Cache for a week
        )

        if not data:
            return []

        return data if isinstance(data, list) else data.get("events", [])

    def get_historical_rounds(
        self,
        tour: str = "pga",
        event_id: str = "all",
        year: int = 2024
    ) -> List[Dict[str, Any]]:
        """
        Get round-level scoring, stats, strokes-gained for historical events.

        Args:
            tour: Tour code (pga, euro, liv, etc.)
            event_id: Event ID or "all" for all events
            year: 1983-2026 depending on tour

        Returns round-by-round data including:
        - Score, strokes gained by category
        - Traditional stats (fairways, greens, putts)
        - Tee times
        """
        data = self._request(
            "/historical-raw-data/rounds",
            params={
                "tour": tour,
                "event_id": event_id,
                "year": str(year),
                "file_format": "json"
            },
            cache_hours=168  # Cache for a week
        )

        if not data:
            return []

        return data if isinstance(data, list) else data.get("rounds", [])

    # =========================================================================
    # NEW ENDPOINTS - Historical Event Stats
    # =========================================================================

    def get_historical_event_finishes(
        self,
        event_id: str,
        year: int = 2025
    ) -> List[Dict[str, Any]]:
        """
        Get event-level finishes, earnings, FedExCup points.

        Args:
            event_id: Event ID from event list
            year: 2025-2026
        """
        data = self._request(
            "/historical-event-data/events",
            params={
                "tour": "pga",
                "event_id": event_id,
                "year": str(year),
                "file_format": "json"
            },
            cache_hours=24
        )

        if not data:
            return []

        return data if isinstance(data, list) else data.get("results", [])

    # =========================================================================
    # NEW ENDPOINTS - Historical DFS Data
    # =========================================================================

    def get_dfs_event_list(self) -> List[Dict[str, Any]]:
        """Get list of events with DFS data available."""
        data = self._request(
            "/historical-dfs-data/event-list",
            params={"file_format": "json"},
            cache_hours=168
        )

        if not data:
            return []

        return data if isinstance(data, list) else data.get("events", [])

    def get_dfs_points(
        self,
        tour: str = "pga",
        site: str = "draftkings",
        event_id: str = "all",
        year: int = 2024
    ) -> List[Dict[str, Any]]:
        """
        Get DFS salaries, ownerships, and actual points scored.

        Args:
            tour: pga or euro
            site: draftkings or fanduel
            event_id: Event ID or "all"
            year: 2017-2025
        """
        data = self._request(
            "/historical-dfs-data/points",
            params={
                "tour": tour,
                "site": site,
                "event_id": event_id,
                "year": str(year),
                "file_format": "json"
            },
            cache_hours=168
        )

        if not data:
            return []

        return data if isinstance(data, list) else data.get("dfs_data", [])

    # =========================================================================
    # NEW ENDPOINTS - Historical Betting Odds (Extended)
    # =========================================================================

    def get_historical_outrights(
        self,
        tour: str = "pga",
        event_id: str = "all",
        year: int = 2024,
        market: str = "win",
        book: str = "consensus"
    ) -> List[Dict[str, Any]]:
        """
        Get historical opening/closing odds with bet outcomes.

        Args:
            tour: pga, euro, alt
            event_id: Event ID or "all"
            year: 2019-2025
            market: win, top_5, top_10, top_20, make_cut
            book: consensus or specific book (draftkings, fanduel, etc.)
        """
        data = self._request(
            "/historical-odds/outrights",
            params={
                "tour": tour,
                "event_id": event_id,
                "year": str(year),
                "market": market,
                "book": book,
                "odds_format": "percent",
                "file_format": "json"
            },
            cache_hours=168
        )

        if not data:
            return []

        return data if isinstance(data, list) else data.get("odds", [])

    def get_historical_matchups(
        self,
        tour: str = "pga",
        event_id: str = "all",
        year: int = 2024,
        book: str = "consensus"
    ) -> List[Dict[str, Any]]:
        """
        Get historical matchup and 3-ball odds with outcomes.

        Args:
            tour: pga, euro, alt
            event_id: Event ID or "all"
            year: 2019-2025
            book: consensus or specific book
        """
        data = self._request(
            "/historical-odds/matchups",
            params={
                "tour": tour,
                "event_id": event_id,
                "year": str(year),
                "book": book,
                "odds_format": "percent",
                "file_format": "json"
            },
            cache_hours=168
        )

        if not data:
            return []

        return data if isinstance(data, list) else data.get("matchups", [])

    # =========================================================================
    # UTILITY - Get All Available Data for a Golfer
    # =========================================================================

    def get_golfer_full_profile(self, golfer_name: str) -> Dict[str, Any]:
        """
        Get comprehensive profile for a golfer combining multiple endpoints.

        Returns:
            - Skill ratings (SG categories)
            - Approach skill by yardage
            - Current predictions
            - DG ranking
            - Historical performance
        """
        profile = {
            "name": golfer_name,
            "skill_ratings": {},
            "approach_skill": {},
            "current_predictions": {},
            "ranking": {},
            "course_fits": {}
        }

        # Get skill ratings
        skill_ratings = self.get_skill_ratings()
        if golfer_name in skill_ratings:
            profile["skill_ratings"] = skill_ratings[golfer_name]

        # Get approach skill
        approach_skills = self.get_approach_skill()
        if golfer_name in approach_skills:
            buckets = approach_skills[golfer_name]
            profile["approach_skill"] = {
                "50_100": buckets.sg_50_100,
                "100_150": buckets.sg_100_150,
                "150_200": buckets.sg_150_200,
                "200_plus": buckets.sg_200_plus,
                "fairway": buckets.sg_fairway,
                "rough": buckets.sg_rough
            }

        # Get DG ranking
        rankings = self.get_dg_rankings()
        if golfer_name in rankings:
            profile["ranking"] = rankings[golfer_name]

        # Get current predictions
        predictions = self.get_pre_tournament_predictions()
        for pred in predictions:
            if pred.golfer_name == golfer_name:
                profile["current_predictions"] = {
                    "win": pred.win_prob,
                    "top_5": pred.top_5_prob,
                    "top_10": pred.top_10_prob,
                    "top_20": pred.top_20_prob,
                    "make_cut": pred.make_cut_prob
                }
                break

        # Get course fit predictions
        course_fits = self.get_course_fit_predictions()
        if golfer_name in course_fits:
            profile["course_fits"] = {"current": course_fits[golfer_name]}

        return profile


def get_api() -> DataGolfAPI:
    """Get configured API client."""
    return DataGolfAPI()


# =============================================================================
# Weather API - Open-Meteo (Free, no API key required)
# =============================================================================

class WeatherAPI:
    """
    Weather forecast API using Open-Meteo (free, no API key needed).
    Provides wind speed, temperature, precipitation for golf course locations.
    """

    BASE_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self):
        self._session = requests.Session()

    def get_forecast(
        self,
        latitude: float,
        longitude: float,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Get weather forecast for a location.

        Args:
            latitude: Course latitude (WGS84)
            longitude: Course longitude (WGS84)
            days: Number of forecast days (1-16)

        Returns:
            Dict with hourly forecasts including wind, temp, precipitation
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "hourly": "temperature_2m,wind_speed_10m,wind_gusts_10m,wind_direction_10m,precipitation_probability,precipitation",
            "daily": "temperature_2m_max,temperature_2m_min,wind_speed_10m_max,wind_gusts_10m_max,precipitation_sum,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": min(days, 16),
        }

        try:
            response = self._session.get(self.BASE_URL, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Weather API error: {e}")
            return {}

    def get_tournament_weather(
        self,
        latitude: float,
        longitude: float,
        tournament_dates: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get weather forecast formatted for golf tournament analysis.

        Returns:
            Dict with daily summaries and scoring impact estimates
        """
        data = self.get_forecast(latitude, longitude, days=10)

        if not data or "daily" not in data:
            return {}

        daily = data.get("daily", {})
        times = daily.get("time", [])

        forecasts = []
        for i, date in enumerate(times):
            wind_max = daily.get("wind_speed_10m_max", [0] * len(times))[i]
            wind_gust = daily.get("wind_gusts_10m_max", [0] * len(times))[i]
            precip = daily.get("precipitation_sum", [0] * len(times))[i]
            precip_prob = daily.get("precipitation_probability_max", [0] * len(times))[i]
            temp_max = daily.get("temperature_2m_max", [20] * len(times))[i]
            temp_min = daily.get("temperature_2m_min", [10] * len(times))[i]

            # Calculate scoring difficulty adjustment
            # Wind: every 10 mph over 10 mph adds ~0.3 strokes
            wind_mph = wind_max * 0.621371  # km/h to mph
            wind_impact = max(0, (wind_mph - 10) / 10 * 0.3)

            # Rain: precipitation reduces scoring, adds ~0.2 strokes per 5mm
            rain_impact = (precip / 5) * 0.2 if precip > 0 else 0

            # Temperature extremes affect scoring
            temp_impact = 0
            if temp_max > 35:  # Very hot
                temp_impact = 0.15
            elif temp_min < 5:  # Very cold
                temp_impact = 0.2

            total_impact = wind_impact + rain_impact + temp_impact

            forecasts.append({
                "date": date,
                "wind_max_mph": round(wind_mph, 1),
                "wind_gust_mph": round(wind_gust * 0.621371, 1),
                "precipitation_mm": round(precip, 1),
                "precipitation_probability": precip_prob,
                "temp_high_c": round(temp_max, 1),
                "temp_low_c": round(temp_min, 1),
                "scoring_impact": round(total_impact, 2),
                "conditions": self._describe_conditions(wind_mph, precip, precip_prob),
            })

        return {
            "location": {
                "latitude": latitude,
                "longitude": longitude,
            },
            "forecasts": forecasts,
        }

    def _describe_conditions(self, wind_mph: float, precip: float, precip_prob: int) -> str:
        """Generate human-readable conditions description."""
        conditions = []

        if wind_mph >= 25:
            conditions.append("Very windy")
        elif wind_mph >= 15:
            conditions.append("Windy")
        elif wind_mph >= 10:
            conditions.append("Breezy")
        else:
            conditions.append("Calm")

        if precip > 5 or precip_prob > 70:
            conditions.append("Rain likely")
        elif precip > 0 or precip_prob > 40:
            conditions.append("Possible showers")

        return ", ".join(conditions)


def get_weather_api() -> WeatherAPI:
    """Get weather API client."""
    return WeatherAPI()


# =============================================================================
# The Odds API - Betting odds aggregation (requires API key)
# =============================================================================

class OddsAPI:
    """
    Betting odds aggregation from The Odds API.
    Provides consensus odds from DraftKings, FanDuel, BetMGM, etc.
    Free tier: 500 requests/month.
    Get API key at: https://the-odds-api.com/
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: Optional[str] = None):
        config = get_config()
        self.api_key = api_key or getattr(config, 'odds_api_key', None) or os.getenv("ODDS_API_KEY")
        self._session = requests.Session()

    def is_configured(self) -> bool:
        """Check if API key is configured."""
        return bool(self.api_key)

    def get_golf_sports(self) -> List[Dict[str, str]]:
        """
        Get list of available golf betting markets.
        Returns sports keys like 'golf_pga_championship_winner'.
        """
        if not self.api_key:
            logger.warning("ODDS_API_KEY not configured")
            return []

        try:
            response = self._session.get(
                f"{self.BASE_URL}/sports",
                params={"apiKey": self.api_key},
                timeout=15
            )
            response.raise_for_status()
            sports = response.json()

            # Filter for golf
            golf_sports = [
                {"key": s["key"], "title": s["title"], "active": s["active"]}
                for s in sports
                if "golf" in s["key"].lower()
            ]
            return golf_sports
        except requests.RequestException as e:
            logger.error(f"Odds API error: {e}")
            return []

    def get_tournament_odds(
        self,
        sport_key: str = None,
        regions: str = "us",
        markets: str = "outrights"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Get betting odds for a golf tournament.

        Args:
            sport_key: e.g., 'golf_pga_championship_winner' (if None, finds active golf)
            regions: 'us', 'uk', 'eu', 'au'
            markets: 'outrights' for tournament winner

        Returns:
            Dict of golfer_name -> {
                consensus_prob: average implied probability,
                best_odds: best available odds,
                books: {book_name: odds}
            }
        """
        if not self.api_key:
            logger.warning("ODDS_API_KEY not configured")
            return {}

        # Find active golf sport if not specified
        if not sport_key:
            golf_sports = self.get_golf_sports()
            active_golf = [s for s in golf_sports if s.get("active")]
            if not active_golf:
                logger.info("No active golf tournaments found")
                return {}
            sport_key = active_golf[0]["key"]

        try:
            response = self._session.get(
                f"{self.BASE_URL}/sports/{sport_key}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": regions,
                    "markets": markets,
                    "oddsFormat": "american",
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            if not data:
                return {}

            # Parse odds from all bookmakers
            golfer_odds = {}

            for event in data:
                for bookmaker in event.get("bookmakers", []):
                    book_name = bookmaker.get("key", "unknown")

                    for market in bookmaker.get("markets", []):
                        if market.get("key") != "outrights":
                            continue

                        for outcome in market.get("outcomes", []):
                            name = outcome.get("name", "")
                            price = outcome.get("price", 0)

                            if not name:
                                continue

                            # Convert American odds to implied probability
                            if price > 0:
                                implied_prob = 100 / (price + 100)
                            else:
                                implied_prob = abs(price) / (abs(price) + 100)

                            if name not in golfer_odds:
                                golfer_odds[name] = {
                                    "books": {},
                                    "prices": [],
                                    "probs": [],
                                }

                            golfer_odds[name]["books"][book_name] = {
                                "price": price,
                                "implied_prob": round(implied_prob, 4),
                            }
                            golfer_odds[name]["prices"].append(price)
                            golfer_odds[name]["probs"].append(implied_prob)

            # Calculate consensus for each golfer
            result = {}
            for name, data in golfer_odds.items():
                probs = data["probs"]
                prices = data["prices"]

                result[name] = {
                    "consensus_prob": round(sum(probs) / len(probs), 4) if probs else 0,
                    "best_odds": max(prices) if prices else 0,
                    "worst_odds": min(prices) if prices else 0,
                    "num_books": len(data["books"]),
                    "books": data["books"],
                }

            logger.info(f"Fetched odds for {len(result)} golfers from {sport_key}")
            return result

        except requests.RequestException as e:
            logger.error(f"Odds API error: {e}")
            return {}

    def compare_to_model(
        self,
        model_probs: Dict[str, float],
        market_odds: Dict[str, Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Compare model probabilities to market odds to find value.

        Args:
            model_probs: Dict of golfer_name -> win probability from model
            market_odds: Market odds (fetched if not provided)

        Returns:
            List of value plays sorted by edge, with:
            - name, model_prob, market_prob, edge, best_odds
        """
        if market_odds is None:
            market_odds = self.get_tournament_odds()

        if not market_odds:
            return []

        value_plays = []
        for name, model_prob in model_probs.items():
            if name not in market_odds:
                continue

            market_data = market_odds[name]
            market_prob = market_data.get("consensus_prob", 0)

            if market_prob <= 0:
                continue

            # Edge = model probability - market implied probability
            edge = model_prob - market_prob

            value_plays.append({
                "name": name,
                "model_prob": round(model_prob, 4),
                "market_prob": round(market_prob, 4),
                "edge": round(edge, 4),
                "edge_pct": round(edge / market_prob * 100, 1) if market_prob > 0 else 0,
                "best_odds": market_data.get("best_odds", 0),
                "num_books": market_data.get("num_books", 0),
            })

        # Sort by edge descending
        value_plays.sort(key=lambda x: x["edge"], reverse=True)
        return value_plays

    # =========================================================================
    # Premium API Features
    # =========================================================================

    def get_historical_odds(
        self,
        sport_key: str,
        event_id: str = None,
        date: str = None,
        markets: str = "outrights"
    ) -> Dict[str, Any]:
        """
        Get historical odds (Premium feature).

        Args:
            sport_key: e.g., 'golf_pga_championship_winner'
            event_id: Specific event ID (optional)
            date: ISO date string for historical snapshot
            markets: Market type

        Returns:
            Historical odds data
        """
        if not self.api_key:
            return {}

        try:
            params = {
                "apiKey": self.api_key,
                "regions": "us",
                "markets": markets,
                "oddsFormat": "american",
            }
            if date:
                params["date"] = date

            url = f"{self.BASE_URL}/historical/sports/{sport_key}/odds"
            if event_id:
                url = f"{self.BASE_URL}/historical/sports/{sport_key}/events/{event_id}/odds"

            response = self._session.get(url, params=params, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Historical odds API error: {e}")
            return {}

    def get_event_scores(self, sport_key: str = None, days_from: int = 3) -> List[Dict]:
        """
        Get scores/results for completed events (Premium feature).

        Args:
            sport_key: Sport key (finds active golf if None)
            days_from: Number of days to look back

        Returns:
            List of event scores/results
        """
        if not self.api_key:
            return []

        if not sport_key:
            golf_sports = self.get_golf_sports()
            if golf_sports:
                sport_key = golf_sports[0]["key"]
            else:
                return []

        try:
            response = self._session.get(
                f"{self.BASE_URL}/sports/{sport_key}/scores",
                params={
                    "apiKey": self.api_key,
                    "daysFrom": days_from,
                },
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Scores API error: {e}")
            return []

    def get_head_to_head_odds(self, sport_key: str = None) -> Dict[str, Any]:
        """
        Get head-to-head matchup odds (Premium feature).

        Returns:
            Dict with matchup odds from various books
        """
        if not self.api_key:
            return {}

        if not sport_key:
            golf_sports = self.get_golf_sports()
            active = [s for s in golf_sports if s.get("active")]
            if not active:
                return {}
            sport_key = active[0]["key"]

        try:
            response = self._session.get(
                f"{self.BASE_URL}/sports/{sport_key}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": "us",
                    "markets": "h2h",
                    "oddsFormat": "american",
                },
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"H2H odds API error: {e}")
            return {}

    def get_prop_odds(self, sport_key: str = None, market: str = "player_top_5") -> Dict[str, Any]:
        """
        Get prop bet odds (Premium feature).

        Args:
            sport_key: Sport key
            market: Prop market type (player_top_5, player_top_10, player_top_20, etc.)

        Returns:
            Dict with prop odds
        """
        if not self.api_key:
            return {}

        if not sport_key:
            golf_sports = self.get_golf_sports()
            active = [s for s in golf_sports if s.get("active")]
            if not active:
                return {}
            sport_key = active[0]["key"]

        try:
            response = self._session.get(
                f"{self.BASE_URL}/sports/{sport_key}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": "us",
                    "markets": market,
                    "oddsFormat": "american",
                },
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            # Parse prop odds
            prop_odds = {}
            for event in data:
                for bookmaker in event.get("bookmakers", []):
                    book = bookmaker.get("key", "unknown")
                    for mkt in bookmaker.get("markets", []):
                        for outcome in mkt.get("outcomes", []):
                            name = outcome.get("name", "")
                            price = outcome.get("price", 0)
                            point = outcome.get("point")

                            if name not in prop_odds:
                                prop_odds[name] = {"books": {}}

                            prob = 100 / (price + 100) if price > 0 else abs(price) / (abs(price) + 100)
                            prop_odds[name]["books"][book] = {
                                "price": price,
                                "implied_prob": round(prob, 4),
                                "point": point
                            }

            # Add consensus
            for name, data in prop_odds.items():
                probs = [b["implied_prob"] for b in data["books"].values()]
                data["consensus_prob"] = round(sum(probs) / len(probs), 4) if probs else 0
                data["num_books"] = len(data["books"])

            return prop_odds
        except requests.RequestException as e:
            logger.error(f"Prop odds API error: {e}")
            return {}

    def get_api_usage(self) -> Dict[str, Any]:
        """
        Check API usage/quota (included in response headers).
        """
        if not self.api_key:
            return {"error": "No API key configured"}

        try:
            response = self._session.get(
                f"{self.BASE_URL}/sports",
                params={"apiKey": self.api_key},
                timeout=10
            )
            return {
                "requests_remaining": response.headers.get("x-requests-remaining"),
                "requests_used": response.headers.get("x-requests-used"),
            }
        except requests.RequestException as e:
            return {"error": str(e)}


def get_odds_api() -> OddsAPI:
    """Get odds API client."""
    return OddsAPI()
