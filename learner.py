"""
Learning Engine for PGA One and Done Optimizer.

Provides continuous learning capabilities:
1. Outcome Tracking - Record predictions vs actual results
2. Course Fit Calibration - Learn which skills correlate with success at each course
3. Opponent Pattern Learning - Track opponent preferences and predict picks
4. Elite Tier Auto-Updates - Dynamically adjust elite tier assignments
5. Model Confidence Scoring - Track prediction accuracy with Brier scores
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple
import math

try:
    from .database import Database
    from .config import get_config
except ImportError:
    from database import Database
    from config import get_config


# Static elite tier definitions (matching strategy.py)
TIER_1_ELITE = {
    "Scottie Scheffler", "Rory McIlroy", "Jon Rahm",
    "Xander Schauffele", "Viktor Hovland"
}
TIER_2_ELITE = {
    "Patrick Cantlay", "Collin Morikawa", "Ludvig Aberg",
    "Wyndham Clark", "Brian Harman", "Tommy Fleetwood",
    "Sam Burns", "Matt Fitzpatrick", "Tom Kim", "Max Homa"
}


@dataclass
class LearningInsight:
    """A single insight from the learning system."""
    category: str  # 'course_fit', 'elite_tier', 'opponent', 'accuracy'
    title: str
    description: str
    confidence: float  # 0-1
    impact: str  # 'high', 'medium', 'low'
    data: Dict  # Additional structured data


class OutcomeTracker:
    """
    Track predictions before tournaments and compare with actual outcomes.
    Records all recommendations so we can learn from results.
    """

    def __init__(self, db: Database = None):
        self.db = db or Database()

    def record_prediction(
        self,
        golfer_name: str,
        tournament_name: str,
        tournament_date: date,
        purse: int,
        win_prob: float,
        top10_prob: float,
        expected_value: float,
        course_fit: float = 0,
        strategic_score: float = 0,
        was_save_warning: bool = False,
        is_my_pick: bool = False
    ):
        """
        Record a prediction before a tournament starts.
        Call this when generating recommendations.
        """
        self.db.save_pick_outcome(
            golfer_name=golfer_name,
            tournament_name=tournament_name,
            tournament_date=tournament_date,
            purse=purse,
            predicted_win_prob=win_prob,
            predicted_top10_prob=top10_prob,
            predicted_ev=expected_value,
            predicted_course_fit=course_fit,
            strategic_score=strategic_score,
            was_save_warning=was_save_warning,
            was_my_pick=is_my_pick
        )

    def record_batch_predictions(
        self,
        tournament_name: str,
        tournament_date: date,
        purse: int,
        recommendations: List[Dict],
        my_pick_name: str = None
    ):
        """Record multiple predictions at once (e.g., top 20 recommendations)."""
        for rec in recommendations:
            self.record_prediction(
                golfer_name=rec.get("golfer_name", rec.get("golfer", {}).get("name", "")),
                tournament_name=tournament_name,
                tournament_date=tournament_date,
                purse=purse,
                win_prob=rec.get("win_prob", 0),
                top10_prob=rec.get("top_10_prob", 0),
                expected_value=rec.get("expected_value", 0),
                course_fit=rec.get("course_fit", 0),
                strategic_score=rec.get("strategic_score", 0),
                was_save_warning=rec.get("was_save_warning", False),
                is_my_pick=(rec.get("golfer_name") == my_pick_name)
            )

    def record_outcome(
        self,
        golfer_name: str,
        tournament_name: str,
        actual_position: int,
        actual_earnings: int,
        made_cut: bool
    ) -> bool:
        """
        Record the actual outcome for a golfer.
        Call this after tournament completes.
        """
        return self.db.record_pick_outcome_result(
            golfer_name=golfer_name,
            tournament_name=tournament_name,
            actual_position=actual_position,
            actual_earnings=actual_earnings,
            made_cut=made_cut
        )

    def get_pending_outcomes(self) -> List[Dict]:
        """Get predictions that need outcome recording."""
        return self.db.get_pending_outcomes()

    def get_outcome_summary(self, days: int = 365) -> Dict:
        """
        Get summary statistics of prediction accuracy.
        """
        outcomes = self.db.get_pick_outcomes(days=days, recorded_only=True)
        if not outcomes:
            return {
                "total_predictions": 0,
                "avg_ev_error": 0,
                "ev_correlation": 0,
                "my_picks_performance": {}
            }

        # Calculate EV accuracy
        ev_errors = []
        my_pick_outcomes = []

        for o in outcomes:
            if o.get("actual_earnings") is not None and o.get("predicted_ev"):
                error = o["predicted_ev"] - o["actual_earnings"]
                ev_errors.append(error)

            if o.get("was_my_pick"):
                my_pick_outcomes.append(o)

        avg_ev_error = sum(ev_errors) / len(ev_errors) if ev_errors else 0

        # My picks performance
        my_picks = {
            "total": len(my_pick_outcomes),
            "wins": sum(1 for o in my_pick_outcomes if o.get("actual_position") == 1),
            "top10s": sum(1 for o in my_pick_outcomes if o.get("actual_position") and o["actual_position"] <= 10),
            "cuts_made": sum(1 for o in my_pick_outcomes if o.get("made_cut")),
            "total_earnings": sum(o.get("actual_earnings", 0) for o in my_pick_outcomes),
            "total_predicted_ev": sum(o.get("predicted_ev", 0) for o in my_pick_outcomes),
        }

        return {
            "total_predictions": len(outcomes),
            "avg_ev_error": avg_ev_error,
            "my_picks_performance": my_picks
        }


class CourseFitLearner:
    """
    Learn which skills actually correlate with success at each course.
    Adjusts static course fit weights based on real tournament outcomes.
    """

    # Base skill weights (these get adjusted)
    SKILL_NAMES = ["sg_putt", "sg_app", "sg_arg", "sg_ott", "driving_acc"]

    def __init__(self, db: Database = None):
        self.db = db or Database()

    def update_course_fit_weights(
        self,
        tournament_name: str,
        outcomes: List[Dict],
        static_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Update course fit weights based on tournament outcomes.

        Args:
            tournament_name: Name of the tournament
            outcomes: List of golfer outcomes with actual position/earnings
            static_weights: The original static weights for this course

        Returns:
            Dict of updated weights blending static and learned
        """
        if len(outcomes) < 10:
            # Not enough data to learn from
            return static_weights

        # Calculate correlation between each skill and actual performance
        # For now, we just store the static weights until we have more data
        existing = self.db.get_learned_course_fits(tournament_name)
        sample_size = len(existing[0]["sample_size"]) if existing else 0

        for skill_name in self.SKILL_NAMES:
            static_weight = static_weights.get(skill_name, 0.2)

            # Blend static and learned (more weight to learned as sample grows)
            confidence = min(0.8, sample_size / 50)  # Max 80% confidence
            learned_weight = static_weight  # Placeholder - would calculate from outcomes

            self.db.save_learned_course_fit(
                tournament_name=tournament_name,
                skill_name=skill_name,
                static_weight=static_weight,
                learned_weight=learned_weight,
                confidence=confidence,
                sample_size=sample_size + len(outcomes)
            )

        return self.get_blended_weights(tournament_name, static_weights)

    def get_blended_weights(
        self,
        tournament_name: str,
        static_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Get course fit weights blending static and learned based on confidence.
        """
        learned = self.db.get_learned_course_fits(tournament_name)
        if not learned:
            return static_weights

        blended = {}
        for entry in learned:
            skill = entry["skill_name"]
            confidence = entry.get("confidence", 0)
            static = entry.get("static_weight", static_weights.get(skill, 0.2))
            learned_w = entry.get("learned_weight", static)

            # Blend: more confident = more weight to learned
            blended[skill] = (1 - confidence) * static + confidence * learned_w

        return blended

    def get_learning_status(self) -> List[Dict]:
        """Get current learning status for all courses."""
        all_fits = self.db.get_learned_course_fits()

        # Group by tournament
        by_tournament = {}
        for fit in all_fits:
            name = fit["tournament_name"]
            if name not in by_tournament:
                by_tournament[name] = {
                    "tournament_name": name,
                    "skills": [],
                    "avg_confidence": 0,
                    "total_sample_size": 0
                }
            by_tournament[name]["skills"].append(fit)

        # Calculate averages
        result = []
        for name, data in by_tournament.items():
            if data["skills"]:
                data["avg_confidence"] = sum(s["confidence"] for s in data["skills"]) / len(data["skills"])
                data["total_sample_size"] = max(s["sample_size"] for s in data["skills"])
            result.append(data)

        return sorted(result, key=lambda x: x["avg_confidence"], reverse=True)


class OpponentPatternLearner:
    """
    Learn opponent picking patterns to predict their likely picks.
    Identifies contrarian opportunities.
    """

    def __init__(self, db: Database = None):
        self.db = db or Database()

    def analyze_opponent(
        self,
        opponent_name: str,
        picks: List[Dict],
        golfer_stats: Dict[str, Dict]
    ) -> Dict:
        """
        Analyze an opponent's picking history to learn their patterns.

        Args:
            opponent_name: Name of the opponent
            picks: List of their historical picks with golfer info
            golfer_stats: Dict of golfer stats keyed by name

        Returns:
            Dict of learned patterns
        """
        if not picks:
            return {}

        # Analyze their preferences
        rankings = []
        win_probs = []

        for pick in picks:
            golfer_name = pick.get("golfer_name")
            if golfer_name and golfer_name in golfer_stats:
                stats = golfer_stats[golfer_name]
                if stats.get("owgr"):
                    rankings.append(stats["owgr"])
                if stats.get("win_prob"):
                    win_probs.append(stats["win_prob"])

        # Calculate preference scores
        avg_ranking = sum(rankings) / len(rankings) if rankings else 100
        avg_win_prob = sum(win_probs) / len(win_probs) if win_probs else 0.02

        # Prefers favorites if avg ranking < 30
        prefers_favorites = max(0, min(1, 1 - (avg_ranking - 10) / 80))

        # Risk tolerance based on variance in picks
        risk_tolerance = 0.5  # Default moderate

        pattern = {
            "prefers_favorites": prefers_favorites,
            "prefers_value": 1 - prefers_favorites,
            "risk_tolerance": risk_tolerance,
            "avg_golfer_ranking": avg_ranking,
            "avg_win_prob_selected": avg_win_prob,
            "total_picks_tracked": len(picks)
        }

        # Save to database
        self.db.save_opponent_pattern(
            opponent_name=opponent_name,
            **pattern
        )

        return pattern

    def predict_likely_picks(
        self,
        opponent_name: str,
        available_golfers: List[Dict],
        tournament_name: str
    ) -> List[Tuple[str, float]]:
        """
        Predict which golfers an opponent is likely to pick.

        Returns:
            List of (golfer_name, probability) tuples sorted by likelihood
        """
        pattern = self.db.get_opponent_pattern(opponent_name)
        if not pattern:
            return []

        predictions = []
        for golfer in available_golfers:
            name = golfer.get("name", "")
            owgr = golfer.get("owgr", 100)
            win_prob = golfer.get("win_prob", 0.01)

            # Score based on pattern match
            score = 0.5  # Base probability

            # Adjust based on preferences
            if pattern["prefers_favorites"] > 0.6 and owgr < 30:
                score += 0.2
            elif pattern["prefers_favorites"] < 0.4 and owgr > 50:
                score += 0.1

            predictions.append((name, score))

        return sorted(predictions, key=lambda x: x[1], reverse=True)

    def get_contrarian_opportunities(
        self,
        field: List[Dict],
        opponent_patterns: List[Dict]
    ) -> List[Dict]:
        """
        Identify golfers that opponents are unlikely to pick.
        """
        # Calculate aggregate prediction for each golfer
        golfer_likelihood = {}

        for golfer in field:
            name = golfer.get("name", "")
            # Start with base likelihood from ownership data
            likelihood = golfer.get("ownership_pct", 10) / 100

            golfer_likelihood[name] = {
                "golfer": golfer,
                "expected_ownership": likelihood,
                "is_contrarian": likelihood < 0.15
            }

        contrarian = [
            g for g in golfer_likelihood.values()
            if g["is_contrarian"] and g["golfer"].get("win_prob", 0) > 0.01
        ]

        return sorted(contrarian, key=lambda x: x["golfer"].get("win_prob", 0), reverse=True)


class EliteTierManager:
    """
    Dynamically manage elite tier assignments based on season performance.
    Promotes/demotes players from elite tiers.
    """

    def __init__(self, db: Database = None):
        self.db = db or Database()

    def get_static_tier(self, golfer_name: str) -> int:
        """Get the static tier for a golfer (0=not elite, 1=tier1, 2=tier2)."""
        if golfer_name in TIER_1_ELITE:
            return 1
        elif golfer_name in TIER_2_ELITE:
            return 2
        return 0

    def update_tier(
        self,
        golfer_name: str,
        wins: int = 0,
        top10s: int = 0,
        events_played: int = 0,
        total_earnings: float = 0
    ):
        """
        Update a golfer's tier based on their season performance.
        """
        static_tier = self.get_static_tier(golfer_name)

        # Calculate performance score
        # Wins worth 100 pts, top10s worth 20 pts, earnings worth 1 pt per $100k
        performance_score = (wins * 100) + (top10s * 20) + (total_earnings / 100000)

        # Determine learned tier
        if performance_score >= 200:  # Elite performance
            learned_tier = 1
        elif performance_score >= 80:  # Strong performance
            learned_tier = 2
        elif performance_score >= 30:  # Moderate
            learned_tier = 3 if static_tier > 0 else 0
        else:
            learned_tier = static_tier  # Keep static

        # Confidence based on events played
        tier_confidence = min(0.9, events_played / 15)

        self.db.save_learned_elite_tier(
            golfer_name=golfer_name,
            static_tier=static_tier,
            learned_tier=learned_tier,
            performance_score=performance_score,
            tier_confidence=tier_confidence,
            wins_this_season=wins,
            top10s_this_season=top10s,
            events_played=events_played,
            total_earnings=total_earnings
        )

    def get_effective_tier(self, golfer_name: str) -> int:
        """
        Get the effective tier for a golfer (blending static and learned).
        """
        tiers = self.db.get_learned_elite_tiers()
        for t in tiers:
            if t["golfer_name"] == golfer_name:
                confidence = t.get("tier_confidence", 0)
                if confidence > 0.5:
                    return t["learned_tier"]
                return t["static_tier"]

        return self.get_static_tier(golfer_name)

    def get_tier_changes(self) -> List[Dict]:
        """Get all golfers whose learned tier differs from static."""
        return self.db.get_elite_tier_changes()

    def get_promotion_candidates(self) -> List[Dict]:
        """Get golfers who should be promoted to a higher tier."""
        all_tiers = self.db.get_learned_elite_tiers()
        return [
            t for t in all_tiers
            if t["learned_tier"] < t["static_tier"] or
               (t["static_tier"] == 0 and t["learned_tier"] > 0)
        ]

    def get_demotion_candidates(self) -> List[Dict]:
        """Get golfers who should be demoted to a lower tier."""
        all_tiers = self.db.get_learned_elite_tiers()
        return [
            t for t in all_tiers
            if t["learned_tier"] > t["static_tier"] and t["static_tier"] > 0
        ]


class ModelConfidenceTracker:
    """
    Track model prediction accuracy over time.
    Calculates Brier scores and calibration metrics.
    """

    def __init__(self, db: Database = None):
        self.db = db or Database()

    def calculate_brier_score(self, predictions: List[Tuple[float, bool]]) -> float:
        """
        Calculate Brier score for probability predictions.
        Lower is better (0 = perfect, 1 = worst).

        Args:
            predictions: List of (predicted_probability, actual_outcome) tuples
        """
        if not predictions:
            return 0.5

        total = 0
        for prob, outcome in predictions:
            actual = 1.0 if outcome else 0.0
            total += (prob - actual) ** 2

        return total / len(predictions)

    def track_win_probability_accuracy(
        self,
        tournament_name: str,
        predictions: List[Dict]
    ) -> float:
        """
        Track accuracy of win probability predictions.
        Returns Brier score.
        """
        # Format: (predicted_win_prob, did_win)
        data = [
            (p.get("predicted_win_prob", 0), p.get("actual_position") == 1)
            for p in predictions
            if p.get("actual_position") is not None
        ]

        brier = self.calculate_brier_score(data)

        self.db.save_model_accuracy(
            metric_name="win_prob_brier",
            metric_value=brier,
            tournament_name=tournament_name,
            sample_size=len(data)
        )

        return brier

    def track_top10_probability_accuracy(
        self,
        tournament_name: str,
        predictions: List[Dict]
    ) -> float:
        """Track accuracy of top-10 probability predictions."""
        data = [
            (p.get("predicted_top10_prob", 0), p.get("actual_position", 999) <= 10)
            for p in predictions
            if p.get("actual_position") is not None
        ]

        brier = self.calculate_brier_score(data)

        self.db.save_model_accuracy(
            metric_name="top10_prob_brier",
            metric_value=brier,
            tournament_name=tournament_name,
            sample_size=len(data)
        )

        return brier

    def track_ev_accuracy(
        self,
        tournament_name: str,
        predictions: List[Dict]
    ) -> Tuple[float, float]:
        """
        Track accuracy of expected value predictions.
        Returns (mean_absolute_error, mean_signed_error).
        """
        errors = []
        for p in predictions:
            if p.get("actual_earnings") is not None and p.get("predicted_ev"):
                error = p["predicted_ev"] - p["actual_earnings"]
                errors.append(error)

        if not errors:
            return 0, 0

        mae = sum(abs(e) for e in errors) / len(errors)
        mse = sum(e for e in errors) / len(errors)

        self.db.save_model_accuracy(
            metric_name="ev_mae",
            metric_value=mae,
            tournament_name=tournament_name,
            sample_size=len(errors)
        )

        self.db.save_model_accuracy(
            metric_name="ev_bias",
            metric_value=mse,
            tournament_name=tournament_name,
            sample_size=len(errors)
        )

        return mae, mse

    def get_confidence_trend(self, days: int = 180) -> Dict:
        """
        Get trend of model confidence over time.
        Returns dict with metric trends.
        """
        accuracy_history = self.db.get_model_accuracy_history(days=days)

        # Group by metric
        metrics = {}
        for entry in accuracy_history:
            name = entry["metric_name"]
            if name not in metrics:
                metrics[name] = []
            metrics[name].append({
                "value": entry["metric_value"],
                "date": entry["recorded_at"],
                "sample_size": entry.get("sample_size", 0)
            })

        # Calculate trends
        trends = {}
        for name, values in metrics.items():
            if len(values) >= 2:
                recent = sum(v["value"] for v in values[-5:]) / min(5, len(values))
                older = sum(v["value"] for v in values[:5]) / min(5, len(values))
                trend = "improving" if recent < older else "declining" if recent > older else "stable"
            else:
                trend = "insufficient_data"

            trends[name] = {
                "current": values[-1]["value"] if values else None,
                "trend": trend,
                "history": values
            }

        return trends


class LearningEngine:
    """
    Main learning engine that coordinates all learning components.
    """

    def __init__(self, db: Database = None):
        self.db = db or Database()
        self.outcome_tracker = OutcomeTracker(self.db)
        self.course_fit_learner = CourseFitLearner(self.db)
        self.opponent_learner = OpponentPatternLearner(self.db)
        self.elite_tier_manager = EliteTierManager(self.db)
        self.confidence_tracker = ModelConfidenceTracker(self.db)

    def record_recommendations(
        self,
        tournament_name: str,
        tournament_date: date,
        purse: int,
        recommendations: List[Dict],
        my_pick_name: str = None
    ):
        """
        Record all recommendations for a tournament.
        Should be called when recommendations are generated.
        """
        self.outcome_tracker.record_batch_predictions(
            tournament_name=tournament_name,
            tournament_date=tournament_date,
            purse=purse,
            recommendations=recommendations,
            my_pick_name=my_pick_name
        )

        # Also save individual predictions for accuracy tracking
        for rec in recommendations[:20]:  # Top 20 for tracking
            golfer_name = rec.get("golfer_name", rec.get("golfer", {}).get("name", ""))
            if golfer_name:
                self.db.save_prediction(
                    tournament_name=tournament_name,
                    golfer_name=golfer_name,
                    prediction_type="win",
                    predicted_value=rec.get("win_prob", 0)
                )
                self.db.save_prediction(
                    tournament_name=tournament_name,
                    golfer_name=golfer_name,
                    prediction_type="top10",
                    predicted_value=rec.get("top_10_prob", 0)
                )

    def record_tournament_results(
        self,
        tournament_name: str,
        results: List[Dict]
    ):
        """
        Record actual tournament results and update all learning models.

        Args:
            tournament_name: Name of the tournament
            results: List of dicts with golfer_name, position, earnings, made_cut
        """
        # Record outcomes
        for result in results:
            self.outcome_tracker.record_outcome(
                golfer_name=result["golfer_name"],
                tournament_name=tournament_name,
                actual_position=result["position"],
                actual_earnings=result["earnings"],
                made_cut=result["made_cut"]
            )

            # Record prediction outcomes
            actual_win = 1.0 if result["position"] == 1 else 0.0
            actual_top10 = 1.0 if result["position"] <= 10 else 0.0

            self.db.record_prediction_outcome(
                tournament_name=tournament_name,
                golfer_name=result["golfer_name"],
                prediction_type="win",
                actual_value=actual_win
            )
            self.db.record_prediction_outcome(
                tournament_name=tournament_name,
                golfer_name=result["golfer_name"],
                prediction_type="top10",
                actual_value=actual_top10
            )

            # Update elite tier tracking
            # This would need cumulative stats - simplified here
            self.elite_tier_manager.update_tier(
                golfer_name=result["golfer_name"],
                wins=1 if result["position"] == 1 else 0,
                top10s=1 if result["position"] <= 10 else 0,
                events_played=1,
                total_earnings=result["earnings"]
            )

        # Calculate accuracy metrics
        predictions = self.db.get_pick_outcomes(days=7, recorded_only=True)
        if predictions:
            self.confidence_tracker.track_win_probability_accuracy(tournament_name, predictions)
            self.confidence_tracker.track_top10_probability_accuracy(tournament_name, predictions)
            self.confidence_tracker.track_ev_accuracy(tournament_name, predictions)

    def get_learning_insights(self) -> List[LearningInsight]:
        """
        Generate insights from all learning components.
        """
        insights = []

        # Outcome tracking insights
        summary = self.outcome_tracker.get_outcome_summary(days=90)
        if summary["total_predictions"] > 0:
            my_picks = summary["my_picks_performance"]
            if my_picks.get("total", 0) > 0:
                roi = (my_picks["total_earnings"] - my_picks["total_predicted_ev"]) / max(1, my_picks["total_predicted_ev"])
                insights.append(LearningInsight(
                    category="accuracy",
                    title="Pick Performance",
                    description=f"Your picks: {my_picks['wins']} wins, {my_picks['top10s']} top-10s from {my_picks['total']} picks",
                    confidence=0.8,
                    impact="high" if my_picks["wins"] > 0 else "medium",
                    data=my_picks
                ))

        # Elite tier change insights
        promotions = self.elite_tier_manager.get_promotion_candidates()
        for p in promotions[:3]:
            insights.append(LearningInsight(
                category="elite_tier",
                title=f"Promote: {p['golfer_name']}",
                description=f"Performance score {p['performance_score']:.0f} suggests tier {p['learned_tier']} (was {p['static_tier']})",
                confidence=p.get("tier_confidence", 0.5),
                impact="high",
                data=p
            ))

        demotions = self.elite_tier_manager.get_demotion_candidates()
        for d in demotions[:3]:
            insights.append(LearningInsight(
                category="elite_tier",
                title=f"Demote: {d['golfer_name']}",
                description=f"Performance score {d['performance_score']:.0f} suggests tier {d['learned_tier']} (was {d['static_tier']})",
                confidence=d.get("tier_confidence", 0.5),
                impact="medium",
                data=d
            ))

        # Model confidence insights
        trends = self.confidence_tracker.get_confidence_trend()
        for metric, data in trends.items():
            if data.get("trend") == "improving":
                insights.append(LearningInsight(
                    category="accuracy",
                    title=f"{metric} Improving",
                    description=f"Model {metric} is trending better over time",
                    confidence=0.7,
                    impact="medium",
                    data=data
                ))

        return insights

    def get_dashboard_data(self) -> Dict:
        """
        Get all data needed for the learning insights dashboard.
        """
        return {
            "outcome_summary": self.outcome_tracker.get_outcome_summary(),
            "pending_outcomes": self.outcome_tracker.get_pending_outcomes(),
            "course_fit_status": self.course_fit_learner.get_learning_status(),
            "elite_tier_changes": self.elite_tier_manager.get_tier_changes(),
            "opponent_patterns": self.db.get_opponent_patterns(),
            "model_confidence": self.confidence_tracker.get_confidence_trend(),
            "insights": [
                {
                    "category": i.category,
                    "title": i.title,
                    "description": i.description,
                    "confidence": i.confidence,
                    "impact": i.impact
                }
                for i in self.get_learning_insights()
            ]
        }
