"""
Data Golf API client for PGA One and Done Optimizer.
Fetches golfer predictions, odds, and statistics.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import requests

try:
    from .config import get_config
    from .database import Database
    from .models import Golfer, GolferStats, ApproachBuckets
except ImportError:
    from config import get_config
    from database import Database
    from models import Golfer, GolferStats, ApproachBuckets

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

    def _request(self, endpoint: str, params: Optional[Dict] = None, cache_hours: int = 1) -> Optional[Dict]:
        """Make API request with caching."""
        if not self.api_key:
            logger.warning("No Data Golf API key configured")
            return None

        # Check cache first
        cache_key = f"datagolf:{endpoint}:{str(params)}"
        cached = self.db.get_cache(cache_key)
        if cached:
            logger.debug(f"Using cached data for {endpoint}")
            return cached

        # Make request
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        params["key"] = self.api_key

        try:
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Cache the response
            expires = datetime.now() + timedelta(hours=cache_hours)
            self.db.set_cache(cache_key, data, expires)

            return data
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
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
        baseline_preds = data.get("baseline_history_fit", [])

        for player in baseline_preds:
            pred = PredictionData(
                golfer_name=player.get("player_name", ""),
                datagolf_id=str(player.get("dg_id", "")),
                win_prob=player.get("win_prob", 0),
                top_5_prob=player.get("top_5_prob", 0),
                top_10_prob=player.get("top_10_prob", 0),
                top_20_prob=player.get("top_20_prob", 0),
                make_cut_prob=player.get("make_cut_prob", 0),
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
            params={"tour": tour, "file_format": "json"},
            cache_hours=24
        )

        if not data:
            return {}

        approach_data = {}
        for player in data.get("players", []):
            name = player.get("player_name", "")
            if name:
                approach_data[name] = ApproachBuckets(
                    sg_50_100=player.get("sg_50_100", 0),
                    sg_100_150=player.get("sg_100_150", 0),
                    sg_150_200=player.get("sg_150_200", 0),
                    sg_200_plus=player.get("sg_200_plus", 0),
                    sg_fairway=player.get("sg_fairway", 0),
                    sg_rough=player.get("sg_rough", 0),
                )
        return approach_data

    def get_betting_outrights(self, tour: str = "pga", market: str = "win") -> Dict[str, Dict[str, float]]:
        """
        Get betting odds from multiple sportsbooks.
        market: 'win', 'top_5', 'top_10', 'top_20', 'make_cut'
        Returns dict of golfer_name -> {book: implied_prob, consensus: avg_prob}
        """
        data = self._request(
            "/betting/outrights",
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
            "/betting/matchups",
            params={"tour": tour, "file_format": "json"},
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

    def get_course_fit_predictions(self, tour: str = "pga") -> Dict[str, float]:
        """
        Get course fit adjustments from Data Golf's baseline + course history & fit model.
        Returns strokes gained adjustment per round based on course fit.
        """
        data = self._request(
            "/preds/pre-tournament",
            params={"tour": tour, "file_format": "json"},
            cache_hours=1
        )

        if not data:
            return {}

        fit_adjustments = {}
        # Compare baseline vs baseline_history_fit to get course adjustment
        baseline = {p.get("player_name"): p for p in data.get("baseline", [])}
        baseline_fit = {p.get("player_name"): p for p in data.get("baseline_history_fit", [])}

        for name, fit_data in baseline_fit.items():
            base_data = baseline.get(name, {})
            # Course fit adjustment = difference in expected position
            base_pos = base_data.get("expected_place", 50)
            fit_pos = fit_data.get("expected_place", 50)
            # Convert position difference to approximate SG/round
            # Rough estimate: 1 position = 0.03 SG/round
            fit_adjustments[name] = (base_pos - fit_pos) * 0.03

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
        # Get player list
        players = self.get_player_list()
        if not players:
            logger.warning("No players returned from API")
            return 0

        # Get skill ratings
        skills = self.get_skill_ratings()

        # Get approach skill data
        approach_skills = self.get_approach_skill()

        # Get course fit adjustments
        course_fits = self.get_course_fit_predictions()

        count = 0
        for player in players:
            name = player["name"]
            skill_data = skills.get(name, {})
            approach_data = approach_skills.get(name)
            course_fit = course_fits.get(name, 0)

            golfer = Golfer(
                name=name,
                owgr=player.get("owgr", 999),
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


def get_api() -> DataGolfAPI:
    """Get configured API client."""
    return DataGolfAPI()
