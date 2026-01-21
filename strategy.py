"""
Strategy and recommendation engine for PGA One and Done Optimizer.
Implements Grok's EV-hedged allocation with game theory hedging.
"""

import logging
from datetime import date
from typing import List, Optional, Dict, Tuple
from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans

from .config import get_config, get_schedule, get_next_tournament, get_majors
from .database import Database
from .models import (
    Tournament, Golfer, Recommendation, SeasonPhase, Tier,
    SimulationResult, LeagueStanding
)
from .simulator import Simulator
from .api import DataGolfAPI

logger = logging.getLogger(__name__)


class Strategy:
    """Strategy and recommendation engine."""

    def __init__(self):
        """Initialize strategy engine."""
        self.config = get_config()
        self.db = Database()
        self.simulator = Simulator()
        self.api = DataGolfAPI()

    def get_current_phase(self, target_date: date = None) -> SeasonPhase:
        """Determine current season phase."""
        target_date = target_date or date.today()
        month = target_date.month

        if month <= 3:
            return SeasonPhase.EARLY
        elif month <= 7:
            return SeasonPhase.MID
        else:
            return SeasonPhase.PLAYOFF

    def classify_golfer_tier(self, golfer: Golfer) -> str:
        """Classify golfer by tier based on OWGR."""
        if golfer.owgr <= 20:
            return "elite"
        elif golfer.owgr <= 50:
            return "mid_tier"
        elif golfer.owgr <= 100:
            return "solid"
        else:
            return "longshot"

    def get_opponent_usage_stats(self) -> Dict[str, Dict]:
        """Analyze opponent usage patterns."""
        all_picks = self.db.get_opponent_picks()
        standings = self.db.get_latest_standings()
        league_size = len(standings) if standings else 80

        usage_count = self.db.get_all_golfer_usage()

        stats = {}
        for golfer_name, count in usage_count.items():
            pct_used = (count / league_size) * 100 if league_size > 0 else 0
            stats[golfer_name] = {
                "count": count,
                "pct_used": pct_used,
                "scarcity": 100 - pct_used,  # Higher = more unique if you pick
            }
        return stats

    def calculate_hedge_bonus(self, golfer_name: str, league_size: int = 80) -> float:
        """
        Calculate differentiation bonus for picking an underused golfer.
        Higher bonus = golfer used by fewer opponents.
        """
        usage = self.db.get_golfer_usage_count(golfer_name)
        pct_available = (league_size - usage) / league_size

        # Bonus scales with scarcity
        if pct_available >= 0.95:  # Almost nobody has used
            return 1.15  # 15% bonus
        elif pct_available >= 0.80:
            return 1.08
        elif pct_available >= 0.50:
            return 1.02
        else:
            return 1.0  # No bonus for commonly used golfers

    def calculate_regret_risk(
        self,
        golfer: Golfer,
        tournament: Tournament,
        alternatives: List[Golfer]
    ) -> float:
        """
        Calculate opportunity cost risk of picking this golfer.
        Lower = safer choice.
        """
        if not alternatives:
            return 0

        # Get EV for selected golfer
        selected_ev = self.simulator.calculate_ev(golfer, tournament)

        # Compare to top alternatives
        alt_evs = []
        for alt in alternatives[:3]:
            alt_ev = self.simulator.calculate_ev(alt, tournament)
            alt_evs.append(alt_ev)

        if not alt_evs:
            return 0

        max_alt_ev = max(alt_evs)
        avg_alt_ev = np.mean(alt_evs)

        # Regret = how much we might lose vs best alternative
        regret = max(0, max_alt_ev - selected_ev)

        # Normalize to 0-1 scale (assuming max regret around $500K)
        return min(1.0, regret / 500_000)

    def get_recommendations(
        self,
        tournament: Tournament = None,
        top_n: int = 5,
        available_only: bool = True
    ) -> List[Recommendation]:
        """
        Get ranked recommendations for a tournament.
        Implements Grok's EV-hedged allocation strategy.
        """
        tournament = tournament or get_next_tournament()
        if not tournament:
            logger.error("No upcoming tournament found")
            return []

        logger.info(f"Generating recommendations for {tournament.name}")

        # Get available golfers
        if available_only:
            available_names = self.db.get_available_golfers()
            used_names = set(self.db.get_used_golfers())
        else:
            available_names = [g.name for g in self.db.get_all_golfers()]
            used_names = set()

        # Get golfers with predictions
        golfers = []
        field_golfers = self.api.get_tournament_field_with_predictions(tournament.name)

        for fg in field_golfers:
            if available_only:
                if fg.name in used_names:
                    continue
                if available_names and fg.name not in available_names:
                    continue
            golfers.append(fg)

        if not golfers:
            # Fallback to database golfers
            golfers = [g for g in self.db.get_all_golfers() if g.name not in used_names]

        # Get current phase for strategy adjustment
        phase = self.get_current_phase()
        league_size = len(self.db.get_latest_standings()) or 80

        recommendations = []

        for golfer in golfers:
            # Run simulation
            sim_result = self.simulator.simulate_tournament(golfer, tournament)

            # Calculate EV components
            win_ev = tournament.winner_share * golfer.win_probability
            top_10_ev = tournament.get_payout(5) * golfer.top_10_probability  # Avg top-10 payout
            cut_ev = tournament.get_payout(40) * golfer.make_cut_probability  # Avg make-cut

            # Calculate hedge bonus
            hedge_bonus = self.calculate_hedge_bonus(golfer.name, league_size)

            # Calculate regret risk (vs other available golfers)
            other_golfers = [g for g in golfers if g.name != golfer.name][:5]
            regret_risk = self.calculate_regret_risk(golfer, tournament, other_golfers)

            # Apply phase-specific adjustments
            phase_multiplier = self._get_phase_multiplier(golfer, tournament, phase)

            # Final EV with hedge bonus
            expected_value = sim_result.mean_earnings * hedge_bonus * phase_multiplier

            # Build reasoning
            reasoning = self._build_reasoning(
                golfer, tournament, sim_result, hedge_bonus, phase, regret_risk
            )

            # Confidence score (0-1)
            confidence = self._calculate_confidence(golfer, sim_result)

            rec = Recommendation(
                golfer=golfer,
                tournament=tournament,
                expected_value=expected_value,
                win_ev=win_ev,
                top_10_ev=top_10_ev,
                cut_ev=cut_ev,
                confidence=confidence,
                hedge_bonus=hedge_bonus - 1.0,  # Store as bonus amount
                regret_risk=regret_risk,
                reasoning=reasoning,
            )
            recommendations.append(rec)

        # Sort by total score (EV + hedge - regret penalty)
        recommendations.sort(key=lambda r: r.total_score, reverse=True)

        return recommendations[:top_n]

    def _get_phase_multiplier(
        self,
        golfer: Golfer,
        tournament: Tournament,
        phase: SeasonPhase
    ) -> float:
        """Get strategy multiplier based on season phase."""
        golfer_tier = self.classify_golfer_tier(golfer)

        if phase == SeasonPhase.EARLY:
            # Early season: prefer mid-tier golfers, save elites
            if golfer_tier == "elite":
                if tournament.is_major or tournament.tier == Tier.TIER_1:
                    return 0.85  # Slight penalty for using elite early
                return 0.70  # Bigger penalty for wasting on small events
            elif golfer_tier == "mid_tier":
                return 1.10  # Bonus for using mid-tier appropriately
            else:
                return 1.05

        elif phase == SeasonPhase.MID:
            # Mid season: deploy elites for majors and signatures
            if golfer_tier == "elite":
                if tournament.is_major:
                    return 1.25  # Big bonus for elite at major
                elif tournament.tier == Tier.TIER_1:
                    return 1.15
                return 1.0
            return 1.0

        else:  # PLAYOFF
            # Playoffs: use remaining elites
            if golfer_tier == "elite":
                return 1.20  # Bonus for elite in playoffs
            return 1.0

        return 1.0

    def _calculate_confidence(self, golfer: Golfer, sim_result: SimulationResult) -> float:
        """Calculate confidence score for a recommendation."""
        # Factors: cut rate, consistency (low std), data quality
        cut_factor = sim_result.cut_rate * 0.3
        consistency_factor = max(0, 1 - (sim_result.std_earnings / sim_result.mean_earnings)) * 0.3 if sim_result.mean_earnings > 0 else 0
        rank_factor = max(0, (200 - golfer.owgr) / 200) * 0.4

        return min(1.0, cut_factor + consistency_factor + rank_factor)

    def _build_reasoning(
        self,
        golfer: Golfer,
        tournament: Tournament,
        sim: SimulationResult,
        hedge: float,
        phase: SeasonPhase,
        regret: float
    ) -> str:
        """Build explanation for recommendation."""
        parts = []

        # Performance expectation
        parts.append(f"EV: ${sim.mean_earnings:,.0f}")
        parts.append(f"Win: {sim.win_rate*100:.1f}%")
        parts.append(f"Top-10: {sim.top_10_rate*100:.1f}%")
        parts.append(f"Cut: {sim.cut_rate*100:.0f}%")

        # Tier and phase
        tier = self.classify_golfer_tier(golfer)
        parts.append(f"[{tier.upper()}]")

        # Hedge bonus
        if hedge > 1.05:
            parts.append(f"Hedge: +{(hedge-1)*100:.0f}%")

        # Risk
        if regret > 0.3:
            parts.append("HIGH RISK")
        elif regret < 0.1:
            parts.append("SAFE")

        return " | ".join(parts)

    def analyze_opponent_patterns(self) -> Dict:
        """
        Use ML clustering to analyze opponent strategies.
        Identifies common patterns to exploit.
        """
        opponents = self.db.get_opponent_picks()
        if len(opponents) < 10:
            return {"patterns": [], "message": "Insufficient data for pattern analysis"}

        # Build feature matrix: opponent usage by golfer tier
        opponent_features = defaultdict(lambda: {"elite": 0, "mid": 0, "solid": 0, "longshot": 0})

        for pick in opponents:
            golfer = self.db.get_golfer(pick.golfer_name)
            if golfer:
                tier = self.classify_golfer_tier(golfer)
                if tier == "elite":
                    opponent_features[pick.opponent_username]["elite"] += 1
                elif tier == "mid_tier":
                    opponent_features[pick.opponent_username]["mid"] += 1
                elif tier == "solid":
                    opponent_features[pick.opponent_username]["solid"] += 1
                else:
                    opponent_features[pick.opponent_username]["longshot"] += 1

        if len(opponent_features) < 5:
            return {"patterns": [], "message": "Need more opponent data"}

        # Convert to numpy array
        usernames = list(opponent_features.keys())
        X = np.array([
            [f["elite"], f["mid"], f["solid"], f["longshot"]]
            for f in opponent_features.values()
        ])

        # Cluster into 3 strategy types
        n_clusters = min(3, len(X))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # Analyze clusters
        patterns = []
        for i in range(n_clusters):
            cluster_mask = labels == i
            cluster_mean = X[cluster_mask].mean(axis=0)
            cluster_users = [u for u, l in zip(usernames, labels) if l == i]

            # Determine strategy type
            if cluster_mean[0] > cluster_mean[1]:  # More elites than mid
                strategy_type = "aggressive_elite"
                desc = "Burns elites early"
            elif cluster_mean[3] > cluster_mean[2]:  # More longshots
                strategy_type = "high_variance"
                desc = "Takes risks on longshots"
            else:
                strategy_type = "balanced"
                desc = "Conservative mid-tier focus"

            patterns.append({
                "type": strategy_type,
                "description": desc,
                "count": len(cluster_users),
                "pct": len(cluster_users) / len(usernames) * 100,
                "elite_avg": cluster_mean[0],
                "mid_avg": cluster_mean[1],
            })

        return {
            "patterns": patterns,
            "total_opponents": len(usernames),
            "recommendation": self._pattern_recommendation(patterns),
        }

    def _pattern_recommendation(self, patterns: List[Dict]) -> str:
        """Generate counter-strategy recommendation."""
        if not patterns:
            return "Insufficient data"

        elite_burners = sum(p["count"] for p in patterns if p["type"] == "aggressive_elite")
        total = sum(p["count"] for p in patterns)

        if elite_burners / total > 0.5:
            return "Most opponents burn elites early. SAVE your elites for majors/playoffs."
        elif elite_burners / total < 0.3:
            return "Few opponents using elites. Consider deploying one for a big signature event."
        else:
            return "Mixed strategies. Focus on EV maximization with moderate hedging."

    def get_season_plan(
        self,
        risk_level: int = None,
        remaining_elites: int = 4
    ) -> Dict:
        """
        Generate full season pick plan.
        """
        risk_level = risk_level or self.config.risk_level
        schedule = get_schedule()
        today = date.today()

        # Get used golfers
        used = set(self.db.get_used_golfers())

        # Get all golfers
        all_golfers = self.db.get_all_golfers()
        available = [g for g in all_golfers if g.name not in used]

        # Separate by tier
        elites = [g for g in available if g.owgr <= 20]
        mid_tier = [g for g in available if 20 < g.owgr <= 50]
        solid = [g for g in available if 50 < g.owgr <= 100]

        # Get remaining tournaments
        remaining = [t for t in schedule if t.date >= today]

        # Get majors and signatures
        majors = [t for t in remaining if t.is_major]
        signatures = [t for t in remaining if t.tier == Tier.TIER_1 and not t.is_major]
        regular = [t for t in remaining if t.tier != Tier.TIER_1]

        plan = {
            "strategy": "EV-hedged allocation",
            "risk_level": risk_level,
            "remaining_tournaments": len(remaining),
            "available_elites": len(elites),
            "available_mid_tier": len(mid_tier),
            "recommendations": {
                "majors": [],
                "signatures": [],
                "regular": [],
            },
            "elite_reservation": [],
        }

        # Reserve elites for majors
        if elites and majors:
            for major in majors[:remaining_elites]:
                if elites:
                    best_elite = max(elites, key=lambda g: 1/g.owgr)  # Best by rank
                    plan["elite_reservation"].append({
                        "tournament": major.name,
                        "golfer": best_elite.name,
                        "reasoning": f"Elite reserved for major (OWGR: {best_elite.owgr})"
                    })
                    elites.remove(best_elite)

        # Remaining elites for top signatures
        for sig in signatures[:len(elites)]:
            if elites:
                best = elites.pop(0)
                plan["recommendations"]["signatures"].append({
                    "tournament": sig.name,
                    "golfer": best.name,
                    "tier": "elite",
                })

        # Mid-tier for regular events
        for event in regular:
            if mid_tier:
                best = mid_tier.pop(0)
                plan["recommendations"]["regular"].append({
                    "tournament": event.name,
                    "golfer": best.name,
                    "tier": "mid_tier",
                })
            elif solid:
                best = solid.pop(0)
                plan["recommendations"]["regular"].append({
                    "tournament": event.name,
                    "golfer": best.name,
                    "tier": "solid",
                })

        return plan

    def what_if_pick(self, golfer_name: str, tournament_name: str = None) -> Dict:
        """
        Run what-if analysis for a specific pick.
        """
        from .config import get_tournament_by_name

        tournament = None
        if tournament_name:
            tournament = get_tournament_by_name(tournament_name)
        if not tournament:
            tournament = get_next_tournament()

        if not tournament:
            return {"error": "No tournament found"}

        golfer = self.db.get_golfer(golfer_name)
        if not golfer:
            return {"error": f"Golfer '{golfer_name}' not found"}

        # Run simulation
        sim = self.simulator.simulate_tournament(golfer, tournament)

        # Get alternatives for comparison
        available = self.db.get_available_golfers()
        other_golfers = [
            self.db.get_golfer(n) for n in available
            if n != golfer_name
        ]
        other_golfers = [g for g in other_golfers if g][:5]

        # Regret analysis
        regret = self.simulator.regret_analysis(golfer, tournament, other_golfers)

        return {
            "golfer": golfer_name,
            "tournament": tournament.name,
            "purse": tournament.purse,
            "expected_value": sim.mean_earnings,
            "median_value": sim.median_earnings,
            "win_probability": sim.win_rate,
            "top_10_probability": sim.top_10_rate,
            "cut_probability": sim.cut_rate,
            "upside_90th": sim.percentile_90,
            "downside_10th": sim.percentile_10,
            "regret_analysis": regret,
            "recommendation": "PICK" if regret["regret_risk"] < sim.mean_earnings * 0.2 else "CONSIDER ALTERNATIVES",
        }


def get_strategy() -> Strategy:
    """Get configured strategy engine."""
    return Strategy()
