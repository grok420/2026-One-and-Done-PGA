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

try:
    from .config import get_config, get_schedule, get_next_tournament, get_majors, get_no_cut_events, get_course_profile, CourseProfile
    from .database import Database
    from .models import (
        Tournament, Golfer, Recommendation, SeasonPhase, Tier, CutRule,
        SimulationResult, LeagueStanding, CourseHistory, GolferAvailability
    )
    from .simulator import Simulator
    from .api import DataGolfAPI
except ImportError:
    from config import get_config, get_schedule, get_next_tournament, get_majors, get_no_cut_events, get_course_profile, CourseProfile
    from database import Database
    from models import (
        Tournament, Golfer, Recommendation, SeasonPhase, Tier, CutRule,
        SimulationResult, LeagueStanding, CourseHistory, GolferAvailability
    )
    from simulator import Simulator
    from api import DataGolfAPI

# OWGR threshold for warnings - last year's winner never picked outside top 65
OWGR_WARNING_THRESHOLD = 65

# Tournament value thresholds (purse in millions)
HIGH_VALUE_PURSE = 15_000_000  # Majors, Players, signatures
MID_VALUE_PURSE = 10_000_000   # Strong regular events
BASE_PURSE = 8_000_000         # Standard events

# Field strength thresholds (average OWGR)
WEAK_FIELD_THRESHOLD = 80      # Avg OWGR > 80 = weak field
STRONG_FIELD_THRESHOLD = 40    # Avg OWGR < 40 = strong field

# Make-cut probability threshold for warnings
MAKE_CUT_WARNING_THRESHOLD = 0.80  # Flag golfers with <80% cut probability

# Strategy insight: 65% of season earnings come from best 5 events
# Winners at majors earn $3-4.5M vs $1-2M at regular events

logger = logging.getLogger(__name__)


class Strategy:
    """Strategy and recommendation engine."""

    def __init__(self):
        """Initialize strategy engine."""
        self.config = get_config()
        self.db = Database()
        self.simulator = Simulator()
        self.api = DataGolfAPI()
        # Cached enhanced data from API
        self._cached_decompositions = None
        self._cached_approach_skill = None
        self._cached_dg_rankings = None
        self._cached_betting_odds = None
        self._cached_skill_ratings = None

    def _fetch_enhanced_data(self, tournament: Tournament) -> Dict[str, Any]:
        """
        Fetch ALL available API data for enhanced recommendations.
        Caches data to avoid repeated API calls.
        """
        enhanced_data = {}

        # 1. Player Decompositions - detailed SG breakdown by category
        if self._cached_decompositions is None:
            try:
                self._cached_decompositions = self.api.get_player_decompositions()
                logger.info(f"Fetched decompositions for {len(self._cached_decompositions)} players")
            except Exception as e:
                logger.warning(f"Failed to fetch decompositions: {e}")
                self._cached_decompositions = {}
        enhanced_data['decompositions'] = self._cached_decompositions

        # 2. Approach Skill - SG by yardage bucket and lie
        if self._cached_approach_skill is None:
            try:
                self._cached_approach_skill = self.api.get_approach_skill()
                logger.info(f"Fetched approach skill for {len(self._cached_approach_skill)} players")
            except Exception as e:
                logger.warning(f"Failed to fetch approach skill: {e}")
                self._cached_approach_skill = {}
        enhanced_data['approach_skill'] = self._cached_approach_skill

        # 3. DG Rankings - true skill ratings
        if self._cached_dg_rankings is None:
            try:
                self._cached_dg_rankings = self.api.get_dg_rankings()
                logger.info(f"Fetched DG rankings for {len(self._cached_dg_rankings)} players")
            except Exception as e:
                logger.warning(f"Failed to fetch DG rankings: {e}")
                self._cached_dg_rankings = {}
        enhanced_data['dg_rankings'] = self._cached_dg_rankings

        # 4. Skill Ratings - overall skill estimates
        if self._cached_skill_ratings is None:
            try:
                self._cached_skill_ratings = self.api.get_skill_ratings()
                logger.info(f"Fetched skill ratings for {len(self._cached_skill_ratings)} players")
            except Exception as e:
                logger.warning(f"Failed to fetch skill ratings: {e}")
                self._cached_skill_ratings = {}
        enhanced_data['skill_ratings'] = self._cached_skill_ratings

        # 5. Betting Odds - market consensus (fresh each tournament)
        try:
            self._cached_betting_odds = self.api.get_betting_outrights()
            logger.info(f"Fetched betting odds for {len(self._cached_betting_odds)} players")
        except Exception as e:
            logger.warning(f"Failed to fetch betting odds: {e}")
            self._cached_betting_odds = {}
        enhanced_data['betting_odds'] = self._cached_betting_odds

        return enhanced_data

    def _calculate_sg_course_match(
        self,
        golfer_name: str,
        tournament: Tournament,
        decompositions: Dict,
        approach_skill: Dict
    ) -> Tuple[float, str]:
        """
        Calculate how well a golfer's SG profile matches the course demands.
        Uses detailed decomposition data for precise course fit.

        Returns (adjustment_sg, explanation)
        """
        course_profile = get_course_profile(tournament.course)
        if not course_profile:
            return 0.0, "No course profile available"

        # Get player decomposition
        player_decomp = decompositions.get(golfer_name, {})
        player_approach = approach_skill.get(golfer_name)

        if not player_decomp:
            return 0.0, "No player decomposition data"

        # Calculate weighted course fit based on course profile
        sg_adjustment = 0.0
        factors = []

        # Off-the-tee fit
        if 'sg_ott' in player_decomp:
            ott_sg = player_decomp['sg_ott']
            # Driving distance course weight
            if course_profile.driving_distance > 0.3:
                adjustment = ott_sg * course_profile.driving_distance * 0.5
                sg_adjustment += adjustment
                if abs(adjustment) > 0.05:
                    factors.append(f"OTT {'advantage' if adjustment > 0 else 'disadvantage'} ({adjustment:+.2f})")

        # Approach fit (use detailed buckets if available)
        if player_approach:
            # Long approach (200+)
            if hasattr(player_approach, 'sg_200_plus') and course_profile.approach_long > 0.3:
                adjustment = player_approach.sg_200_plus * course_profile.approach_long * 0.4
                sg_adjustment += adjustment
                if abs(adjustment) > 0.05:
                    factors.append(f"Long approach {'strength' if adjustment > 0 else 'weakness'}")

            # Mid approach (150-200)
            if hasattr(player_approach, 'sg_150_200') and course_profile.approach_mid > 0.3:
                adjustment = player_approach.sg_150_200 * course_profile.approach_mid * 0.4
                sg_adjustment += adjustment

            # Short approach (100-150)
            if hasattr(player_approach, 'sg_100_150') and course_profile.approach_short > 0.3:
                adjustment = player_approach.sg_100_150 * course_profile.approach_short * 0.4
                sg_adjustment += adjustment

        elif 'sg_app' in player_decomp:
            # Fall back to overall approach
            app_sg = player_decomp['sg_app']
            avg_approach_weight = (course_profile.approach_long + course_profile.approach_mid + course_profile.approach_short) / 3
            adjustment = app_sg * avg_approach_weight * 0.4
            sg_adjustment += adjustment
            if abs(adjustment) > 0.05:
                factors.append(f"Approach {'advantage' if adjustment > 0 else 'disadvantage'}")

        # Around-the-green fit
        if 'sg_arg' in player_decomp and course_profile.around_green > 0.3:
            arg_sg = player_decomp['sg_arg']
            adjustment = arg_sg * course_profile.around_green * 0.4
            sg_adjustment += adjustment
            if abs(adjustment) > 0.05:
                factors.append(f"Short game {'strength' if adjustment > 0 else 'weakness'}")

        # Putting fit
        if 'sg_putt' in player_decomp and course_profile.putting > 0.3:
            putt_sg = player_decomp['sg_putt']
            adjustment = putt_sg * course_profile.putting * 0.5
            sg_adjustment += adjustment
            if abs(adjustment) > 0.05:
                factors.append(f"Putting {'advantage' if adjustment > 0 else 'challenge'}")

        explanation = ", ".join(factors) if factors else "Neutral course fit"
        return sg_adjustment, explanation

    def _get_market_consensus_adjustment(
        self,
        golfer_name: str,
        betting_odds: Dict,
        model_win_prob: float
    ) -> Tuple[float, str]:
        """
        Compare model prediction vs market odds to find value.
        If model is higher than market, golfer may be undervalued.

        Returns (multiplier, explanation)
        """
        golfer_odds = betting_odds.get(golfer_name, {})
        if not golfer_odds:
            return 1.0, ""

        market_win_prob = golfer_odds.get('win', 0)
        if market_win_prob <= 0:
            return 1.0, ""

        # Calculate edge: model vs market
        edge = model_win_prob - market_win_prob

        if edge > 0.03:  # Model sees >3% more win probability than market
            return 1.08, f"Model edge vs market (+{edge*100:.1f}%)"
        elif edge < -0.03:  # Market sees golfer as better than model
            return 0.95, f"Market favors more than model ({edge*100:.1f}%)"
        else:
            return 1.0, ""

    def _get_true_skill_adjustment(
        self,
        golfer_name: str,
        dg_rankings: Dict,
        skill_ratings: Dict
    ) -> Tuple[float, str]:
        """
        Use Data Golf's true skill ratings to adjust for OWGR anomalies.
        Some golfers are better/worse than their OWGR suggests.

        Returns (multiplier, explanation)
        """
        dg_rank = dg_rankings.get(golfer_name, {})
        skill = skill_ratings.get(golfer_name, {})

        if not dg_rank and not skill:
            return 1.0, ""

        # Get DG skill estimate
        dg_skill = dg_rank.get('dg_skill_estimate', 0) if dg_rank else 0
        owgr = dg_rank.get('owgr', 999) if dg_rank else 999

        # If DG skill suggests player is better than OWGR indicates
        if dg_skill > 0 and owgr > 30:
            # Player underrated by OWGR
            if dg_skill > 1.5:  # Elite skill but not elite OWGR
                return 1.10, f"DG skill ({dg_skill:.2f}) exceeds OWGR #{owgr}"
            elif dg_skill > 0.8:
                return 1.05, f"Undervalued by OWGR (DG: {dg_skill:.2f})"

        # If OWGR is high but DG skill is low
        if owgr < 30 and dg_skill < 0.5:
            return 0.95, f"OWGR may overrate (DG: {dg_skill:.2f})"

        return 1.0, ""

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

    def calculate_field_strength(self, tournament: Tournament) -> Tuple[float, str, str]:
        """
        Calculate field strength based on average OWGR of tournament field.

        Returns (avg_owgr, strength_category, description).

        Categories:
        - WEAK: Avg OWGR > 80 (opposite-field, smaller events)
        - MODERATE: Avg OWGR 40-80 (standard events)
        - STRONG: Avg OWGR < 40 (majors, signatures)
        """
        # Get field predictions to determine who's playing
        field_golfers = self.api.get_tournament_field_with_predictions(tournament.name)

        if not field_golfers:
            # Estimate based on tournament type
            if tournament.is_major or tournament.is_signature:
                return 35.0, "STRONG", "Elite field expected (major/signature event)"
            elif tournament.is_opposite_field:
                return 90.0, "WEAK", "Weak field expected (opposite-field event)"
            else:
                return 60.0, "MODERATE", "Standard field expected"

        # Calculate average OWGR of field
        owgr_values = [g.owgr for g in field_golfers if g.owgr < 999]
        if not owgr_values:
            return 60.0, "MODERATE", "Unable to determine field strength"

        avg_owgr = sum(owgr_values) / len(owgr_values)

        # Categorize field strength
        if avg_owgr > WEAK_FIELD_THRESHOLD:
            category = "WEAK"
            description = f"Weak field (avg OWGR: {avg_owgr:.0f}) - mid-tier golfers can compete"
        elif avg_owgr < STRONG_FIELD_THRESHOLD:
            category = "STRONG"
            description = f"Strong field (avg OWGR: {avg_owgr:.0f}) - elite competition"
        else:
            category = "MODERATE"
            description = f"Moderate field (avg OWGR: {avg_owgr:.0f})"

        return avg_owgr, category, description

    def get_field_strength_multiplier(
        self,
        golfer: Golfer,
        tournament: Tournament,
        field_strength: str
    ) -> Tuple[float, str]:
        """
        Calculate EV multiplier based on field strength and golfer tier.

        - Weak field: Boost mid-tier (OWGR 30-60) EV by 10-15%
        - Strong field: Slight penalty for non-elites

        Returns (multiplier, explanation).
        """
        golfer_tier = self.classify_golfer_tier(golfer)

        if field_strength == "WEAK":
            # Weak fields favor mid-tier golfers who can compete for wins
            if golfer_tier == "mid_tier":
                return 1.15, "WEAK FIELD: +15% boost for mid-tier value play"
            elif golfer_tier == "solid":
                return 1.10, "WEAK FIELD: +10% boost for solid value play"
            elif golfer_tier == "elite":
                # Elites still good, but opportunity cost matters
                return 1.0, "WEAK FIELD: Elite pick - consider saving for stronger field"
            else:
                return 1.05, "WEAK FIELD: +5% boost for longshot"

        elif field_strength == "STRONG":
            # Strong fields favor elites, penalize lower tiers
            if golfer_tier == "elite":
                return 1.05, "STRONG FIELD: Elite competing against peers"
            elif golfer_tier == "mid_tier":
                return 0.95, "STRONG FIELD: -5% penalty against elite competition"
            else:
                return 0.90, "STRONG FIELD: -10% penalty against elite competition"

        else:  # MODERATE
            return 1.0, ""

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

    def check_owgr_warning(self, golfer: Golfer) -> Tuple[bool, str]:
        """
        Check if golfer has OWGR risk.
        Last year's One and Done winner never picked anyone outside top 65 OWGR.
        Returns (has_warning, warning_message).
        """
        if golfer.owgr > OWGR_WARNING_THRESHOLD:
            return True, f"OWGR {golfer.owgr} is outside top {OWGR_WARNING_THRESHOLD} - historical winners avoid these picks"
        return False, ""

    def check_cut_probability_warning(
        self,
        golfer: Golfer,
        tournament: Tournament
    ) -> Tuple[bool, str, float]:
        """
        Check if golfer has risky make-cut probability.
        Flag golfers with <80% make-cut probability as HIGH RISK.

        Returns (has_warning, warning_message, ev_penalty_multiplier).
        """
        # No cut events don't need this warning
        if not tournament.has_cut:
            return False, "", 1.0

        # Get cut probability from API predictions or golfer data
        probs = self.db.get_golfer_probability(golfer.name, tournament.name)
        if probs:
            cut_prob = probs.get("make_cut_prob", golfer.make_cut_probability)
        else:
            cut_prob = golfer.make_cut_probability

        # If no cut probability available, estimate from OWGR
        if not cut_prob or cut_prob == 0:
            if golfer.owgr <= 10:
                cut_prob = 0.92
            elif golfer.owgr <= 25:
                cut_prob = 0.85
            elif golfer.owgr <= 50:
                cut_prob = 0.78
            elif golfer.owgr <= 100:
                cut_prob = 0.68
            else:
                cut_prob = 0.55

        if cut_prob < MAKE_CUT_WARNING_THRESHOLD:
            # Calculate penalty: the lower the cut probability, the higher the penalty
            # At 70% cut prob: multiply EV by 0.9 (10% penalty)
            # At 60% cut prob: multiply EV by 0.85 (15% penalty)
            # At 50% cut prob: multiply EV by 0.80 (20% penalty)
            penalty = 0.9 - (0.1 * (MAKE_CUT_WARNING_THRESHOLD - cut_prob) / 0.3)
            penalty = max(0.75, min(0.95, penalty))  # Cap between 75% and 95%

            warning_msg = f"CUT RISK: Only {cut_prob*100:.0f}% to make cut (below {MAKE_CUT_WARNING_THRESHOLD*100:.0f}% threshold)"
            return True, warning_msg, penalty

        return False, "", 1.0

    def calculate_course_fit(self, golfer: Golfer, tournament: Tournament) -> Tuple[float, str]:
        """
        Calculate course fit adjustment (strokes gained per round) based on
        golfer skills matching course characteristics.

        Research basis (Data Golf, PGA Tour stats):
        - Approach shots account for ~40% of scoring advantage
        - Driving accounts for ~28%
        - Short game for ~17%
        - Putting for ~15%
        - Long courses favor bombers (+0.19 SG/round per 10 yards above avg)
        - Narrow courses favor accuracy players

        Returns (sg_adjustment, explanation)
        """
        profile = get_course_profile(tournament.course)
        if not profile:
            return 0.0, "No course profile available"

        stats = golfer.stats
        if not stats:
            return 0.0, "No golfer stats available"

        # Calculate fit score based on skill-course matching
        fit_components = []
        total_adjustment = 0.0

        # === DRIVING ===
        # Tour average driving distance ~295 yards
        # Each 10 yards above average = ~0.15-0.20 SG/round at long courses
        if stats.driving_distance > 0:
            dist_above_avg = (stats.driving_distance - 295) / 10
            distance_fit = dist_above_avg * 0.15 * profile.driving_distance
            total_adjustment += distance_fit
            if abs(distance_fit) > 0.05:
                if distance_fit > 0:
                    fit_components.append(f"Distance +{distance_fit:.2f}")
                else:
                    fit_components.append(f"Distance {distance_fit:.2f}")

        # Driving accuracy - tour average ~60%
        if stats.driving_accuracy > 0:
            acc_above_avg = (stats.driving_accuracy - 60) / 10  # Per 10% above average
            accuracy_fit = acc_above_avg * 0.12 * profile.driving_accuracy
            total_adjustment += accuracy_fit
            if abs(accuracy_fit) > 0.05:
                if accuracy_fit > 0:
                    fit_components.append(f"Accuracy +{accuracy_fit:.2f}")
                else:
                    fit_components.append(f"Accuracy {accuracy_fit:.2f}")

        # === APPROACH (40% of scoring - most important) ===
        # Use SG:Approach directly, weighted by course approach profile
        if stats.sg_approach != 0:
            # Weight approach importance: avg of long/mid/short weights
            approach_weight = (profile.approach_long + profile.approach_mid + profile.approach_short) / 3
            approach_fit = stats.sg_approach * 0.5 * approach_weight
            total_adjustment += approach_fit
            if abs(approach_fit) > 0.05:
                if approach_fit > 0:
                    fit_components.append(f"Approach +{approach_fit:.2f}")
                else:
                    fit_components.append(f"Approach {approach_fit:.2f}")

        # Approach by yardage buckets (if available)
        if stats.approach_buckets:
            buckets = stats.approach_buckets
            # Long approach (200+ yards) - important at bomber courses
            if buckets.sg_200_plus != 0 and profile.approach_long > 0.3:
                long_fit = buckets.sg_200_plus * 0.3 * profile.approach_long
                total_adjustment += long_fit
                if abs(long_fit) > 0.05:
                    fit_components.append(f"Long approach {'+' if long_fit > 0 else ''}{long_fit:.2f}")

            # Short approach (100-150) - important at precision courses
            if buckets.sg_100_150 != 0 and profile.approach_short > 0.3:
                short_fit = buckets.sg_100_150 * 0.25 * profile.approach_short
                total_adjustment += short_fit
                if abs(short_fit) > 0.05:
                    fit_components.append(f"Short approach {'+' if short_fit > 0 else ''}{short_fit:.2f}")

        # === SHORT GAME ===
        if stats.sg_around_green != 0:
            arg_fit = stats.sg_around_green * 0.4 * profile.around_green
            total_adjustment += arg_fit
            if abs(arg_fit) > 0.05:
                if arg_fit > 0:
                    fit_components.append(f"Short game +{arg_fit:.2f}")
                else:
                    fit_components.append(f"Short game {arg_fit:.2f}")

        # === PUTTING ===
        if stats.sg_putting != 0:
            putt_fit = stats.sg_putting * 0.35 * profile.putting
            total_adjustment += putt_fit
            if abs(putt_fit) > 0.05:
                if putt_fit > 0:
                    fit_components.append(f"Putting +{putt_fit:.2f}")
                else:
                    fit_components.append(f"Putting {putt_fit:.2f}")

        # === WIND FACTOR ===
        # Players with low ball flight / links experience do better in wind
        # Approximate: accuracy players handle wind better
        if profile.wind_factor > 0.5 and stats.driving_accuracy > 0:
            wind_bonus = ((stats.driving_accuracy - 55) / 15) * 0.1 * profile.wind_factor
            total_adjustment += wind_bonus
            if abs(wind_bonus) > 0.05:
                fit_components.append(f"Wind handling {'+' if wind_bonus > 0 else ''}{wind_bonus:.2f}")

        # === ROUGH PENALTY ===
        # Players who miss fairways suffer more at courses with penal rough
        if profile.rough_penalty > 0.5 and stats.driving_accuracy > 0:
            # Players below average accuracy get penalized at tough rough courses
            rough_impact = ((stats.driving_accuracy - 60) / 20) * 0.15 * profile.rough_penalty
            total_adjustment += rough_impact
            if rough_impact < -0.05:
                fit_components.append(f"Rough penalty {rough_impact:.2f}")

        # Build explanation
        if fit_components:
            explanation = f"{tournament.course}: {', '.join(fit_components)}"
        else:
            explanation = f"{tournament.course}: Neutral fit"

        # Cap adjustment to reasonable range (-0.8 to +0.8 SG/round)
        total_adjustment = max(-0.8, min(0.8, total_adjustment))

        return total_adjustment, explanation

    def calculate_tournament_value_factor(self, tournament: Tournament) -> float:
        """
        Calculate how valuable this tournament is for deploying elite picks.
        Higher value = save your best golfers for these events.

        Key insight: 65% of season earnings come from best 5 events.
        Winners at majors earn $3-4.5M vs $1-2M at regular events.
        Tour Championship winner ~$3.6M (comparable to majors due to starting strokes).
        """
        base_factor = 1.0

        # Purse-based value (bigger purse = more valuable)
        purse_factor = tournament.purse / BASE_PURSE

        # Tournament type bonuses
        if "Tour Championship" in tournament.name:
            # Tour Championship: Winner ~$3.6M (comparable to majors)
            # Starting strokes system affects effective win probability
            type_bonus = 2.0
        elif tournament.is_major:
            # Majors are key events - winner gets $3.6-4.5M
            type_bonus = 2.0
        elif tournament.is_signature:
            # Signature events have $20M purses, winners get $3.6M
            type_bonus = 1.75
        elif tournament.is_playoff:
            # FedEx St. Jude and BMW Championship - $20M purses
            type_bonus = 1.6
        elif tournament.tier == Tier.TIER_1:
            type_bonus = 1.3
        else:
            type_bonus = 1.0

        return base_factor * purse_factor * type_bonus

    def calculate_opposite_field_boost(
        self,
        golfer: Golfer,
        tournament: Tournament
    ) -> Tuple[float, str]:
        """
        Calculate EV boost for opposite-field events.

        Opposite-field events run concurrently with signature events and
        feature weaker fields. Mid-tier golfers (OWGR 30-60) get a 15% boost
        as they can realistically compete for wins without facing elite players.

        Returns (multiplier, explanation).
        """
        if not tournament.is_opposite_field:
            return 1.0, ""

        golfer_tier = self.classify_golfer_tier(golfer)

        # Mid-tier golfers (OWGR 21-50) get the biggest boost
        if 30 <= golfer.owgr <= 60:
            return 1.15, "OPPOSITE FIELD: +15% boost - prime spot for mid-tier value"
        elif golfer_tier == "mid_tier":  # OWGR 21-50
            return 1.12, "OPPOSITE FIELD: +12% boost - good mid-tier opportunity"
        elif golfer_tier == "solid":  # OWGR 51-100
            return 1.10, "OPPOSITE FIELD: +10% boost - solid player in weak field"
        elif golfer_tier == "elite":
            # Elites rarely play opposite-field events, but if they do...
            return 1.0, "OPPOSITE FIELD: Elite in weak field (unusual)"
        else:
            return 1.05, "OPPOSITE FIELD: +5% boost for longshot"

    # =========================================================================
    # Phase 2.1: Course History Methods
    # =========================================================================

    def get_course_history_boost(
        self,
        golfer: Golfer,
        tournament: Tournament
    ) -> Tuple[float, str, Optional[CourseHistory]]:
        """
        Calculate EV boost based on historical performance at the course.

        Weight recent results more heavily (last 2 years count more).
        Returns (multiplier, explanation, course_history_object).
        """
        course_history = self.db.get_course_history(golfer.name, tournament.course)

        if not course_history or course_history.years_played == 0:
            return 1.0, "", None

        # Calculate boost based on historical performance
        boost = 1.0
        reasons = []

        # Past wins at this course are very valuable
        if course_history.wins > 0:
            win_boost = 0.10 + (course_history.wins - 1) * 0.05  # 10% for first win, +5% for each additional
            boost += min(0.25, win_boost)  # Cap at 25%
            reasons.append(f"{course_history.wins} win{'s' if course_history.wins > 1 else ''}")

        # Strong top-5 record
        elif course_history.top_5s >= 2:
            boost += 0.08  # 8% boost for multiple top-5s
            reasons.append(f"{course_history.top_5s} top-5s")

        # Good top-10 record
        elif course_history.top_10s >= 3:
            boost += 0.05  # 5% boost for consistent top-10s
            reasons.append(f"{course_history.top_10s} top-10s")

        # Penalize poor course history
        if course_history.avg_finish > 40 and course_history.years_played >= 3:
            boost -= 0.05  # 5% penalty for consistently poor finishes
            reasons.append(f"avg finish {course_history.avg_finish:.0f}")

        # High cut miss rate at this course
        if course_history.cut_rate < 0.60 and course_history.years_played >= 2:
            boost -= 0.08  # 8% penalty for frequent cut misses
            reasons.append(f"only {course_history.cut_rate*100:.0f}% cuts made")

        # Recent form at course (last 2 years weighted more)
        if course_history.recent_avg_finish > 0:
            if course_history.recent_avg_finish <= 10:
                boost += 0.05  # Recent top-10 average
                reasons.append(f"recent avg: T{course_history.recent_avg_finish:.0f}")
            elif course_history.recent_avg_finish <= 20:
                boost += 0.02  # Recent top-20 average

        # SG performance at course
        if course_history.sg_total_at_course > 0.5:
            boost += 0.03  # 3% boost for strong SG history
            reasons.append(f"+{course_history.sg_total_at_course:.1f} SG")
        elif course_history.sg_total_at_course < -0.5:
            boost -= 0.03  # 3% penalty for poor SG history

        # Build explanation
        if reasons:
            explanation = f"COURSE HISTORY: {course_history.summary}"
        else:
            explanation = ""

        return boost, explanation, course_history

    # =========================================================================
    # Phase 2.2: Golfer Availability Check
    # =========================================================================

    def get_golfer_availability(
        self,
        golfer: Golfer,
        tournament: Tournament
    ) -> Tuple[GolferAvailability, str]:
        """
        Check golfer availability status for a tournament.

        Returns (availability_status, explanation).
        """
        availability = self.db.get_golfer_availability(golfer.name, tournament.name)

        if availability is None:
            # Try to get from API
            self.api.sync_availability_to_db(tournament.name)
            availability = self.db.get_golfer_availability(golfer.name, tournament.name)

        if availability is None:
            return GolferAvailability.UNKNOWN, ""

        explanations = {
            GolferAvailability.CONFIRMED: "CONFIRMED: In official field",
            GolferAvailability.LIKELY: "LIKELY: Expected to play (>75% probability)",
            GolferAvailability.UNLIKELY: "UNLIKELY: May skip (<50% probability)",
            GolferAvailability.OUT: "OUT: Confirmed not playing",
            GolferAvailability.UNKNOWN: "",
        }

        return availability, explanations.get(availability, "")

    def should_save_elite(self, golfer: Golfer, tournament: Tournament) -> Tuple[bool, str]:
        """
        Determine if an elite golfer should be saved for a better tournament.
        Returns (should_save, reason).
        """
        if self.classify_golfer_tier(golfer) != "elite":
            return False, ""

        phase = self.get_current_phase()
        schedule = get_schedule()
        today = tournament.date

        # Get upcoming high-value tournaments
        upcoming_majors = [t for t in schedule if t.date > today and t.is_major]
        upcoming_signatures = [t for t in schedule if t.date > today and t.is_signature]

        tournament_value = self.calculate_tournament_value_factor(tournament)

        # Early season with majors ahead - save elites
        if phase == SeasonPhase.EARLY:
            if upcoming_majors and tournament_value < 1.5:
                return True, f"Save for upcoming majors ({len(upcoming_majors)} remaining)"

        # Mid season - only use elites at majors/signatures
        if phase == SeasonPhase.MID:
            if not tournament.is_major and not tournament.is_signature:
                if upcoming_majors:
                    return True, f"Reserve for majors ({upcoming_majors[0].name} coming up)"

        # If this is a low-value event with elites still needed
        if tournament_value < 1.2 and len(upcoming_majors) + len(upcoming_signatures) > 0:
            return True, "Low-value event - save elite for high-purse tournaments"

        return False, ""

    def calculate_win_probability_value(
        self,
        golfer: Golfer,
        tournament: Tournament,
        sim_result
    ) -> Tuple[float, str]:
        """
        Calculate the value of a golfer's win probability at this tournament.

        Key insight: 65% of winning entries' points come from ~5 events.
        Those are almost always tournament WINS at high-purse events.

        High win probability is worth MORE at high-purse events.
        Wasting high win probability at low-purse events is a strategic mistake.

        Returns (win_value_multiplier, explanation).
        """
        win_prob = sim_result.win_rate
        winner_share = tournament.winner_share

        # Calculate the "win opportunity value" - what this win prob is worth here
        win_opportunity = win_prob * winner_share

        # Compare to what this golfer's win prob would be worth at a major
        # (Assuming similar win probability, which is generous)
        avg_major_winner_share = 4_000_000  # ~$4M for major winners
        major_opportunity = win_prob * avg_major_winner_share

        # Calculate opportunity cost ratio
        opportunity_ratio = win_opportunity / major_opportunity if major_opportunity > 0 else 1.0

        # Win probability tiers and their strategic value
        if win_prob >= 0.15:  # 15%+ win probability = elite
            tier = "ELITE_WIN_PROB"
            # Elite win prob at low-value event is a waste
            if tournament.purse < 12_000_000 and not tournament.is_major:
                multiplier = 0.6 + (opportunity_ratio * 0.4)  # 60-100% based on purse
                explanation = f"HIGH WIN PROB ({win_prob*100:.1f}%) - worth ${win_opportunity:,.0f} here vs ${major_opportunity:,.0f} at major"
            else:
                multiplier = 1.0 + (win_prob * 0.5)  # Bonus for deploying at right time
                explanation = f"DEPLOY HIGH WIN PROB ({win_prob*100:.1f}%) at high-value event"

        elif win_prob >= 0.08:  # 8-15% = strong
            tier = "STRONG_WIN_PROB"
            if tournament.is_major or tournament.is_signature:
                multiplier = 1.0 + (win_prob * 0.3)  # Small bonus at big events
                explanation = f"Good win chance ({win_prob*100:.1f}%) at premium event"
            else:
                multiplier = 1.0
                explanation = f"Solid win chance ({win_prob*100:.1f}%)"

        elif win_prob >= 0.03:  # 3-8% = moderate
            tier = "MODERATE_WIN_PROB"
            multiplier = 1.0
            explanation = f"Moderate win chance ({win_prob*100:.1f}%)"

        else:  # < 3% = longshot
            tier = "LONGSHOT"
            # Longshots are fine at regular events, not at majors
            if tournament.is_major:
                multiplier = 0.85  # Penalty for wasting major on longshot
                explanation = f"LOW WIN PROB ({win_prob*100:.1f}%) - consider saving major slot"
            else:
                multiplier = 1.0
                explanation = f"Longshot ({win_prob*100:.1f}%) - acceptable at regular event"

        return multiplier, explanation

    def calculate_no_cut_ev(self, golfer: Golfer, tournament: Tournament) -> float:
        """
        Calculate EV for no-cut events where everyone gets paid.
        No-cut events favor higher-variance picks since there's a guaranteed floor.
        """
        if tournament.has_cut:
            return 0  # Use standard EV calculation for cut events

        # Get simulation result
        sim_result = self.simulator.simulate_tournament(golfer, tournament)

        # In no-cut events, minimum payout is guaranteed
        min_payout = tournament.min_payout

        # Adjusted EV = max(mean_earnings, min_payout) for the floor
        # But also boost value of high-upside players since no cut risk
        base_ev = max(sim_result.mean_earnings, min_payout)

        # Upside boost: in no-cut events, high-variance is more valuable
        # because there's no downside of $0 (missed cut)
        variance_boost = (sim_result.percentile_90 - sim_result.mean_earnings) / tournament.purse
        upside_adjustment = 1 + (variance_boost * 0.5)  # Up to 50% boost for high variance

        return base_ev * upside_adjustment

    def get_late_season_popularity_weight(self, target_date: date = None) -> float:
        """
        Calculate popularity weighting factor for late season.
        More important to differentiate from opponents as season progresses.
        Returns multiplier for hedge bonus (1.0 to 2.0).
        """
        target_date = target_date or date.today()
        month = target_date.month

        if month >= 8:  # August playoffs
            return 2.0  # Double the hedge bonus importance
        elif month >= 6:  # June-July
            return 1.5
        elif month >= 4:  # April-May (majors season)
            return 1.25
        else:  # Early season
            return 1.0

    def get_standings_strategy(self) -> Tuple[str, float, str]:
        """
        Determine strategy adjustment based on current standings position.

        Research insight: "If you are leading, matching popular high-upside picks
        can protect your position. If you are trailing, you may need to accept
        more risk and fade popular options."

        Returns (strategy_mode, risk_multiplier, explanation)
        """
        standings = self.db.get_latest_standings()
        if not standings:
            return "neutral", 1.0, "No standings data - using balanced strategy"

        config = get_config()
        my_username = config.site_username.lower()

        # Find my position
        my_standing = None
        total_entries = len(standings)
        for s in standings:
            if s.username.lower() == my_username:
                my_standing = s
                break

        if not my_standing:
            return "neutral", 1.0, "Not found in standings - using balanced strategy"

        percentile = (my_standing.rank / total_entries) * 100

        if percentile <= 10:  # Top 10% - PROTECT
            return "protect", 0.8, f"Top 10% (#{my_standing.rank}) - Match popular picks, reduce variance"
        elif percentile <= 25:  # Top 25% - HOLD
            return "hold", 0.95, f"Top 25% (#{my_standing.rank}) - Balanced approach, slight risk reduction"
        elif percentile <= 50:  # Middle - NEUTRAL
            return "neutral", 1.0, f"Middle of pack (#{my_standing.rank}) - Standard EV maximization"
        elif percentile <= 75:  # Bottom half - AGGRESSIVE
            return "aggressive", 1.15, f"Bottom half (#{my_standing.rank}) - Need upside, fade popular picks"
        else:  # Bottom 25% - DESPERATION
            return "desperation", 1.3, f"Bottom 25% (#{my_standing.rank}) - High-risk required, contrarian picks"

    # Known LIV Golf players (only eligible for majors)
    LIV_GOLFERS = {
        "Rahm, Jon", "DeChambeau, Bryson", "Koepka, Brooks", "Johnson, Dustin",
        "Reed, Patrick", "Mickelson, Phil", "Garcia, Sergio", "Poulter, Ian",
        "Westwood, Lee", "Stenson, Henrik", "Smith, Cameron", "Gooch, Talor",
        "Niemann, Joaquin", "Ancer, Abraham", "Ortiz, Carlos", "Wolff, Matthew",
        "Swafford, Hudson", "Na, Kevin", "Finau, Tony", "Hatton, Tyrrell"
    }

    def is_liv_golfer(self, golfer_name: str) -> bool:
        """Check if golfer plays on LIV Golf (only available at majors)."""
        return golfer_name in self.LIV_GOLFERS

    def check_liv_warning(self, golfer: Golfer, tournament: Tournament) -> Tuple[bool, str]:
        """
        Check if picking a LIV golfer at a non-major.

        Research insight: "The only opportunities to use LIV golfers like
        Jon Rahm and Bryson DeChambeau is at the four majors."

        Returns (is_warning, warning_message)
        """
        if not self.is_liv_golfer(golfer.name):
            return False, ""

        if tournament.is_major:
            return False, ""  # OK to use LIV golfers at majors

        return True, f"LIV golfer - only available at majors (save for Masters, PGA, US Open, Open Championship)"

    def get_season_win_targets(self) -> Dict:
        """
        Track progress toward winning blueprint: 4-6 winners, double top-5s.

        Research insight: "4-6 winners with close to double Top 5 finishes
        was the 2025 Blueprint."
        """
        picks = self.db.get_all_picks()

        wins = 0
        top_5s = 0
        top_10s = 0
        total_earnings = 0

        for pick in picks:
            if pick.position:
                if pick.position == 1:
                    wins += 1
                if pick.position <= 5:
                    top_5s += 1
                if pick.position <= 10:
                    top_10s += 1
            total_earnings += pick.earnings

        # Calculate targets and progress
        target_wins = 5  # Middle of 4-6 range
        target_top_5s = wins * 2 + 4  # Double wins + buffer

        return {
            "wins": wins,
            "top_5s": top_5s,
            "top_10s": top_10s,
            "total_earnings": total_earnings,
            "target_wins": target_wins,
            "target_top_5s": target_top_5s,
            "wins_on_track": wins >= 0,  # Will be calculated based on schedule progress
            "top_5s_on_track": top_5s >= wins * 2,
            "analysis": self._analyze_win_progress(wins, top_5s, total_earnings)
        }

    def _analyze_win_progress(self, wins: int, top_5s: int, earnings: int) -> str:
        """Analyze season progress toward winning targets."""
        schedule = get_schedule()
        today = date.today()
        completed = len([t for t in schedule if t.date < today])
        total = len(schedule)
        progress_pct = (completed / total * 100) if total > 0 else 0

        # Expected wins at this point (target 5 for season)
        expected_wins = (progress_pct / 100) * 5

        if wins >= expected_wins + 1:
            status = "AHEAD"
            advice = "Strong position - can take slightly safer picks to protect lead"
        elif wins >= expected_wins:
            status = "ON TRACK"
            advice = "Maintain balanced strategy - continue targeting high-EV picks"
        elif wins >= expected_wins - 1:
            status = "SLIGHTLY BEHIND"
            advice = "Need a breakthrough - consider higher-upside picks at big events"
        else:
            status = "BEHIND"
            advice = "Aggressive strategy needed - prioritize win probability at remaining majors/signatures"

        return f"{status}: {wins} wins ({progress_pct:.0f}% through season). {advice}"

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

    def calculate_hedge_bonus(
        self,
        golfer_name: str,
        league_size: int = 80,
        target_date: date = None
    ) -> float:
        """
        Calculate differentiation bonus for picking an underused golfer.
        Higher bonus = golfer used by fewer opponents.
        Bonus increases in importance as season progresses.
        """
        usage = self.db.get_golfer_usage_count(golfer_name)
        pct_available = (league_size - usage) / league_size

        # Base bonus scales with scarcity
        if pct_available >= 0.95:  # Almost nobody has used
            base_bonus = 1.15  # 15% bonus
        elif pct_available >= 0.80:
            base_bonus = 1.08
        elif pct_available >= 0.50:
            base_bonus = 1.02
        else:
            base_bonus = 1.0  # No bonus for commonly used golfers

        # Apply late-season multiplier (hedge becomes more important)
        late_season_mult = self.get_late_season_popularity_weight(target_date)
        # Scale the bonus portion by the multiplier
        bonus_portion = base_bonus - 1.0
        adjusted_bonus = 1.0 + (bonus_portion * late_season_mult)

        return adjusted_bonus

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

    def calculate_future_opportunity_value(
        self,
        golfer: Golfer,
        current_tournament: Tournament,
        top_n_futures: int = 5
    ) -> Tuple[float, float, str, List[Dict]]:
        """
        Calculate golfer's maximum EV across future tournaments to determine opportunity cost.

        This is the KEY One and Done insight:
        - A golfer's value at THIS tournament should be compared to their BEST future opportunity
        - If Scheffler has $80K EV this week but $200K EV at The Masters, using him now
          has an opportunity cost of $120K

        Returns:
            - max_future_ev: Highest projected EV at any future tournament
            - opportunity_cost: max_future_ev - current_ev (negative = good to use now)
            - best_future_event: Name of the best future event for this golfer
            - future_evs: List of top future opportunities with details
        """
        schedule = get_schedule()
        today = date.today()

        # Get remaining tournaments after current one
        remaining = [
            t for t in schedule
            if t.date > current_tournament.date
        ]

        if not remaining:
            return 0.0, 0.0, "", []

        # Calculate current EV
        current_ev = self.simulator.calculate_ev(golfer, current_tournament)

        # Calculate EV at each future tournament
        future_opportunities = []

        for future_t in remaining:
            # Quick EV estimate based on purse and golfer skill
            # Use OWGR-based win probability estimate
            # Top 10 golfer: ~8-15% win prob at typical event
            # Top 50 golfer: ~2-5% win prob
            # Beyond 50: ~0.5-2% win prob

            if golfer.owgr <= 10:
                base_win_prob = 0.10
            elif golfer.owgr <= 25:
                base_win_prob = 0.05
            elif golfer.owgr <= 50:
                base_win_prob = 0.025
            elif golfer.owgr <= 100:
                base_win_prob = 0.012
            else:
                base_win_prob = 0.005

            # Adjust for tournament tier/field strength
            if future_t.is_major:
                # Majors have strongest fields, but also biggest purses
                tier_mult = 0.85  # Slightly harder to win
                purse_mult = 1.0
            elif future_t.tier == Tier.TIER_1:
                tier_mult = 0.90
                purse_mult = 1.0
            elif future_t.is_opposite_field:
                tier_mult = 1.3  # Easier field
                purse_mult = 1.0
            else:
                tier_mult = 1.0
                purse_mult = 1.0

            adjusted_win_prob = base_win_prob * tier_mult

            # Simple EV = P(win) * winner_share + P(top10) * avg_top10_payout
            # Winner gets ~18% of purse, top 10 avg ~4%
            win_payout = future_t.purse * 0.18
            top10_payout = future_t.purse * 0.04

            # Estimate top 10 prob as ~3x win prob
            top10_prob = min(0.50, adjusted_win_prob * 3)

            future_ev = (adjusted_win_prob * win_payout) + (top10_prob * top10_payout)

            # Course fit adjustment (if we have it)
            course_fit, _ = self.calculate_course_fit(golfer, future_t)
            fit_adjustment = 1.0 + (course_fit * 0.08)
            future_ev *= fit_adjustment

            future_opportunities.append({
                "tournament": future_t.name,
                "date": future_t.date.strftime("%Y-%m-%d"),
                "purse": future_t.purse,
                "is_major": future_t.is_major,
                "tier": future_t.tier.value if hasattr(future_t.tier, 'value') else str(future_t.tier),
                "projected_ev": future_ev,
                "win_prob": adjusted_win_prob,
                "course_fit": course_fit,
            })

        # Sort by projected EV
        future_opportunities.sort(key=lambda x: x["projected_ev"], reverse=True)

        # Get max future EV
        if future_opportunities:
            max_future_ev = future_opportunities[0]["projected_ev"]
            best_future_event = future_opportunities[0]["tournament"]
        else:
            max_future_ev = 0
            best_future_event = ""

        # Opportunity cost = what we give up by using golfer now
        # Positive = we're sacrificing future value
        # Negative = this IS the best opportunity
        opportunity_cost = max_future_ev - current_ev

        return max_future_ev, opportunity_cost, best_future_event, future_opportunities[:top_n_futures]

    def calculate_relative_value(
        self,
        golfer: Golfer,
        tournament: Tournament
    ) -> Tuple[float, str]:
        """
        Calculate the RELATIVE value of picking this golfer at this tournament.

        Relative Value = Current EV / Max Future EV

        - Value > 1.0: This IS their best tournament (use now!)
        - Value 0.8-1.0: Good opportunity (acceptable to use)
        - Value 0.5-0.8: Mediocre opportunity (consider saving)
        - Value < 0.5: Poor opportunity (save for better event)

        Returns:
            - relative_value: Ratio of current EV to best future EV
            - recommendation: "USE NOW", "ACCEPTABLE", "CONSIDER SAVING", "SAVE"
        """
        max_future_ev, opp_cost, best_event, _ = self.calculate_future_opportunity_value(
            golfer, tournament
        )

        current_ev = self.simulator.calculate_ev(golfer, tournament)

        # Handle edge cases
        if max_future_ev <= 0:
            return 1.5, "USE NOW - No better future events"

        if current_ev <= 0:
            return 0.0, "SAVE - No value this week"

        relative_value = current_ev / max_future_ev

        # Determine recommendation
        if relative_value >= 1.0:
            recommendation = f"USE NOW - Best opportunity (no better future event)"
        elif relative_value >= 0.85:
            recommendation = f"ACCEPTABLE - Good value ({relative_value:.0%} of max)"
        elif relative_value >= 0.60:
            recommendation = f"CONSIDER SAVING - Better at {best_event} ({relative_value:.0%} of max)"
        else:
            recommendation = f"SAVE - Much better at {best_event} ({relative_value:.0%} of max)"

        return relative_value, recommendation

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

        # ENHANCED: Fetch all available API data for better recommendations
        enhanced_data = self._fetch_enhanced_data(tournament)
        decompositions = enhanced_data.get('decompositions', {})
        approach_skill = enhanced_data.get('approach_skill', {})
        dg_rankings = enhanced_data.get('dg_rankings', {})
        skill_ratings = enhanced_data.get('skill_ratings', {})
        betting_odds = enhanced_data.get('betting_odds', {})
        logger.info(f"Enhanced data loaded: {len(decompositions)} decomps, {len(betting_odds)} odds")

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

        # Get standings-based strategy adjustment
        standings_mode, standings_risk_mult, standings_explanation = self.get_standings_strategy()
        logger.info(f"Standings strategy: {standings_mode} ({standings_explanation})")

        recommendations = []

        # Calculate tournament value for context
        tournament_value = self.calculate_tournament_value_factor(tournament)
        is_high_value_event = tournament.is_major or tournament.is_signature or tournament.purse >= HIGH_VALUE_PURSE

        # Phase 1.1: Calculate field strength for this tournament
        avg_owgr, field_strength, field_strength_desc = self.calculate_field_strength(tournament)
        logger.info(f"Field strength: {field_strength} ({field_strength_desc})")

        for golfer in golfers:
            # Check for LIV golfer at non-major (skip with warning)
            liv_warning, liv_msg = self.check_liv_warning(golfer, tournament)
            if liv_warning:
                logger.info(f"Skipping {golfer.name}: {liv_msg}")
                continue  # Don't recommend LIV golfers for non-majors

            # Phase 2.2: Check golfer availability
            availability, availability_msg = self.get_golfer_availability(golfer, tournament)
            if availability == GolferAvailability.OUT:
                logger.info(f"Skipping {golfer.name}: Confirmed OUT")
                continue  # Skip golfers confirmed out
            if availability == GolferAvailability.UNLIKELY:
                logger.info(f"Warning for {golfer.name}: {availability_msg}")
                # Don't skip, but will add warning

            # Phase 2.1: Get course history
            course_history_boost, course_history_msg, course_history = self.get_course_history_boost(
                golfer, tournament
            )

            # Run simulation
            sim_result = self.simulator.simulate_tournament(golfer, tournament)

            # Calculate EV components
            win_ev = tournament.winner_share * golfer.win_probability
            top_10_ev = tournament.get_payout(5) * golfer.top_10_probability  # Avg top-10 payout
            cut_ev = tournament.get_payout(40) * golfer.make_cut_probability  # Avg make-cut

            # Calculate hedge bonus with late-season weighting
            hedge_bonus = self.calculate_hedge_bonus(golfer.name, league_size, tournament.date)

            # Calculate regret risk (vs other available golfers)
            other_golfers = [g for g in golfers if g.name != golfer.name][:5]
            regret_risk = self.calculate_regret_risk(golfer, tournament, other_golfers)

            # Apply phase-specific adjustments
            phase_multiplier = self._get_phase_multiplier(golfer, tournament, phase)

            # Calculate expected value based on cut rule
            if not tournament.has_cut:
                # No-cut event: use special EV calculation
                base_ev = self.calculate_no_cut_ev(golfer, tournament)
            else:
                base_ev = sim_result.mean_earnings

            # Apply tournament value adjustment for elite golfers
            # Penalize using elites at low-value events, reward at high-value events
            golfer_tier = self.classify_golfer_tier(golfer)
            elite_save_penalty = 1.0
            if golfer_tier == "elite":
                should_save, save_reason = self.should_save_elite(golfer, tournament)
                if should_save and not is_high_value_event:
                    elite_save_penalty = 0.7  # 30% penalty for wasting elite
                elif is_high_value_event:
                    elite_save_penalty = 1.15  # 15% bonus for proper elite deployment

            # KEY INSIGHT: Win probability matters more than average EV
            # 65% of winning entries' points come from ~5 events (wins at big purses)
            # High win prob at low-purse event = strategic waste
            win_prob_multiplier, win_prob_explanation = self.calculate_win_probability_value(
                golfer, tournament, sim_result
            )

            # Calculate course fit adjustment
            course_fit, course_fit_explanation = self.calculate_course_fit(golfer, tournament)

            # Convert course fit SG/round to EV adjustment
            # Rough estimate: 1 SG/round improvement = ~3 positions better finish
            # 3 positions better = ~$50K-100K more earnings at typical event
            course_fit_ev_factor = 1.0 + (course_fit * 0.08)  # 8% EV adjustment per SG

            # Apply standings-based risk adjustment
            # If trailing: boost high-variance plays
            # If leading: favor safer picks
            standings_adjustment = 1.0
            if standings_mode == "protect" and sim_result.win_rate < 0.05:
                standings_adjustment = 1.05  # Slight boost to safer picks when leading
            elif standings_mode == "aggressive" and sim_result.win_rate > 0.08:
                standings_adjustment = 1.1  # Boost high-upside picks when trailing
            elif standings_mode == "desperation" and sim_result.win_rate > 0.10:
                standings_adjustment = 1.2  # Big boost to win probability when far behind

            # Phase 1.1: Apply field strength multiplier
            field_strength_mult, field_strength_reason = self.get_field_strength_multiplier(
                golfer, tournament, field_strength
            )

            # Phase 1.2: Check cut probability warning
            cut_warning, cut_warning_msg, cut_penalty = self.check_cut_probability_warning(
                golfer, tournament
            )

            # Phase 1.3: Apply opposite-field boost
            opposite_field_mult, opposite_field_reason = self.calculate_opposite_field_boost(
                golfer, tournament
            )

            # NEW: Calculate RELATIVE VALUE (opportunity cost adjustment)
            # This is the KEY One and Done insight:
            # Compare golfer's EV here vs their BEST future opportunity
            relative_value, relative_recommendation = self.calculate_relative_value(
                golfer, tournament
            )

            # Convert relative value to multiplier
            # relative_value >= 1.0: This IS their best event, boost EV
            # relative_value < 1.0: Better opportunities exist, penalize EV
            # Scale: 0.5 relative value = 0.7x multiplier (30% penalty)
            #        1.0 relative value = 1.0x multiplier (no change)
            #        1.5 relative value = 1.15x multiplier (15% boost)
            if relative_value >= 1.0:
                relative_value_mult = 1.0 + (min(relative_value - 1.0, 0.5) * 0.3)  # Up to 15% boost
            else:
                # Penalize picks that waste golfer on suboptimal events
                # 0.5 relative = 0.7x, 0.8 relative = 0.88x
                relative_value_mult = 0.4 + (relative_value * 0.6)

            # ENHANCED: Detailed SG course matching using decomposition data
            sg_course_match, sg_match_explanation = self._calculate_sg_course_match(
                golfer.name, tournament, decompositions, approach_skill
            )
            # Convert SG match to multiplier (0.5 SG advantage = ~8% EV boost)
            sg_match_mult = 1.0 + (sg_course_match * 0.16)

            # ENHANCED: Market consensus adjustment (model vs betting odds)
            market_mult, market_explanation = self._get_market_consensus_adjustment(
                golfer.name, betting_odds, golfer.win_probability
            )

            # ENHANCED: True skill adjustment (DG skill vs OWGR)
            skill_mult, skill_explanation = self._get_true_skill_adjustment(
                golfer.name, dg_rankings, skill_ratings
            )

            # Final score combines:
            # - Base EV (simulation mean)
            # - Win probability value (don't waste high win% at small events)
            # - Hedge bonus (differentiation from opponents)
            # - Phase multiplier (save elites early, deploy late)
            # - Elite deployment factor
            # - Course fit adjustment
            # - Standings-based risk adjustment
            # - Field strength adjustment (Phase 1.1)
            # - Cut probability penalty (Phase 1.2)
            # - Opposite-field boost (Phase 1.3)
            # - Course history boost (Phase 2.1)
            # - RELATIVE VALUE (opportunity cost)
            # - ENHANCED: SG course match (detailed decomposition)
            # - ENHANCED: Market consensus (model vs odds)
            # - ENHANCED: True skill (DG skill vs OWGR)
            expected_value = (base_ev * win_prob_multiplier * hedge_bonus * phase_multiplier *
                            elite_save_penalty * course_fit_ev_factor * standings_adjustment *
                            field_strength_mult * cut_penalty * opposite_field_mult *
                            course_history_boost * relative_value_mult *
                            sg_match_mult * market_mult * skill_mult)

            # Check OWGR warning
            owgr_warning, owgr_msg = self.check_owgr_warning(golfer)

            # Build reasoning (include OWGR warning and elite save advice if applicable)
            reasoning = self._build_reasoning(
                golfer, tournament, sim_result, hedge_bonus, phase, regret_risk
            )

            # Phase 1.2: Add cut probability warning
            if cut_warning:
                reasoning = f"HIGH RISK: {cut_warning_msg} | {reasoning}"

            if owgr_warning:
                reasoning = f"WARNING: {owgr_msg} | {reasoning}"

            # Phase 1.1: Add field strength context
            if field_strength_reason:
                reasoning = f"{field_strength_reason} | {reasoning}"

            # Phase 1.3: Add opposite-field context
            if opposite_field_reason:
                reasoning = f"{opposite_field_reason} | {reasoning}"

            # Phase 2.1: Add course history context
            if course_history_msg:
                reasoning = f"{course_history_msg} | {reasoning}"

            # Phase 2.2: Add availability warning
            if availability == GolferAvailability.UNLIKELY:
                reasoning = f"AVAILABILITY WARNING: {availability_msg} | {reasoning}"

            # Add win probability strategic advice for high win% golfers
            if sim_result.win_rate >= 0.10:  # 10%+ win probability
                if not is_high_value_event:
                    reasoning = f"WIN PROB ALERT: {win_prob_explanation} | {reasoning}"
                else:
                    reasoning = f"GOOD DEPLOYMENT: {win_prob_explanation} | {reasoning}"

            # Add elite deployment advice
            if golfer_tier == "elite":
                should_save, save_reason = self.should_save_elite(golfer, tournament)
                if should_save:
                    reasoning = f"CONSIDER SAVING: {save_reason} | {reasoning}"
                elif is_high_value_event:
                    reasoning = f"DEPLOY NOW: High-value event (${tournament.purse/1_000_000:.0f}M purse) | {reasoning}"

            # Confidence score (0-1) - reduce for OWGR warning
            confidence = self._calculate_confidence(golfer, sim_result)
            if owgr_warning:
                confidence *= 0.8  # 20% confidence penalty for OWGR risk

            # Add course fit to reasoning if significant
            if abs(course_fit) >= 0.1:
                fit_sign = "+" if course_fit > 0 else ""
                reasoning = f"COURSE FIT: {fit_sign}{course_fit:.2f} SG/rd ({course_fit_explanation}) | {reasoning}"

            # Add RELATIVE VALUE context (opportunity cost)
            if relative_value < 0.7:
                reasoning = f"⚠️ SAVE: {relative_recommendation} | {reasoning}"
            elif relative_value < 0.85:
                reasoning = f"CONSIDER SAVING: {relative_recommendation} | {reasoning}"
            elif relative_value >= 1.0:
                reasoning = f"✓ OPTIMAL: {relative_recommendation} | {reasoning}"

            # Add standings context if adjusting strategy
            if standings_adjustment != 1.0:
                if standings_mode == "protect":
                    reasoning = f"STANDINGS: Protecting lead - favor consistency | {reasoning}"
                elif standings_mode in ("aggressive", "desperation"):
                    reasoning = f"STANDINGS: Trailing - need upside ({standings_adjustment:.0%} boost) | {reasoning}"

            # Get best future event name for display
            _, _, best_future_event, _ = self.calculate_future_opportunity_value(
                golfer, tournament, top_n_futures=1
            )

            # NEW: Generate plain English reasoning
            plain_english_bullets = self._generate_plain_english_bullets(
                golfer=golfer,
                tournament=tournament,
                sim=sim_result,
                course_fit=course_fit,
                relative_value=relative_value,
                best_future_event=best_future_event,
                field_strength=field_strength,
                course_history_summary=course_history.summary if course_history else "",
                # ENHANCED: Add new data insights
                sg_match_explanation=sg_match_explanation,
                market_explanation=market_explanation,
                skill_explanation=skill_explanation,
            )

            # NEW: Calculate factor contributions (additive model)
            factor_contributions = self._calculate_factor_contributions(
                base_ev=base_ev,
                course_fit=course_fit,
                timing_mult=relative_value_mult,
                field_mult=field_strength_mult,
                hedge_mult=hedge_bonus,
                phase_mult=phase_multiplier,
                course_history_boost=course_history_boost,
                # ENHANCED: Add new factor contributions
                sg_match_mult=sg_match_mult,
                market_mult=market_mult,
                skill_mult=skill_mult,
            )

            # NEW: Calculate timing verdict
            timing_verdict = self._calculate_timing_verdict(
                relative_value=relative_value,
                best_future_event=best_future_event,
            )

            # NEW: Calculate confidence percentage
            confidence_pct = self._calculate_confidence_pct(
                golfer=golfer,
                sim=sim_result,
                course_history=course_history,
                availability_status=availability.value if availability else "",
            )

            # NEW: Generate risk flags
            risk_flags = self._generate_risk_flags(
                golfer=golfer,
                sim=sim_result,
                owgr_warning=owgr_warning,
                cut_warning=cut_warning,
                availability_status=availability.value if availability else "",
            )

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
                course_fit_sg=course_fit,
                owgr_warning=owgr_warning,
                # Phase 1 additions
                cut_warning=cut_warning,
                field_strength=field_strength,
                is_opposite_field=tournament.is_opposite_field,
                # Phase 2 additions
                course_history_summary=course_history.summary if course_history else "",
                availability_status=availability.value if availability else "",
                # Opportunity cost / relative value
                relative_value=relative_value,
                best_future_event=best_future_event,
                # NEW: Plain English reasoning fields
                plain_english_bullets=plain_english_bullets,
                factor_contributions=factor_contributions,
                timing_verdict=timing_verdict,
                confidence_pct=confidence_pct,
                risk_flags=risk_flags,
                base_ev=base_ev,
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

        # Cut info varies by event type
        if not tournament.has_cut:
            parts.append(f"NO-CUT (Min: ${tournament.min_payout:,.0f})")
        else:
            parts.append(f"Cut: {sim.cut_rate*100:.0f}%")

        # Tournament value indicator
        tournament_value = self.calculate_tournament_value_factor(tournament)
        if tournament.is_major:
            parts.append("MAJOR")
        elif tournament.is_signature:
            parts.append("SIGNATURE")
        elif tournament_value >= 1.5:
            parts.append("HIGH-VALUE")

        # Tier and phase
        tier = self.classify_golfer_tier(golfer)
        parts.append(f"[{tier.upper()}]")

        # Course fit
        if golfer.course_fit_adjustment != 0:
            fit_dir = "+" if golfer.course_fit_adjustment > 0 else ""
            parts.append(f"Fit: {fit_dir}{golfer.course_fit_adjustment:.2f} SG/rd")

        # Hedge bonus
        if hedge > 1.05:
            parts.append(f"Hedge: +{(hedge-1)*100:.0f}%")

        # Risk
        if regret > 0.3:
            parts.append("HIGH RISK")
        elif regret < 0.1:
            parts.append("SAFE")

        return " | ".join(parts)

    def _generate_plain_english_bullets(
        self,
        golfer: Golfer,
        tournament: Tournament,
        sim: SimulationResult,
        course_fit: float,
        relative_value: float,
        best_future_event: str,
        field_strength: str,
        course_history_summary: str,
        # ENHANCED: New API data insights
        sg_match_explanation: str = "",
        market_explanation: str = "",
        skill_explanation: str = "",
    ) -> List[str]:
        """Generate 3-5 plain English bullet points explaining the pick."""
        bullets = []

        # 1. Course fit reasoning (enhanced with SG decomposition)
        if sg_match_explanation:
            bullets.append(f"Course fit analysis: {sg_match_explanation}")
        elif course_fit >= 0.3:
            bullets.append(f"Excellent course fit at {tournament.course} (+{course_fit:.2f} SG/round)")
        elif course_fit >= 0.1:
            bullets.append(f"Good course fit at {tournament.course} (+{course_fit:.2f} SG/round)")
        elif course_fit <= -0.2:
            bullets.append(f"Poor course fit at {tournament.course} ({course_fit:.2f} SG/round)")

        # 2. Timing/opportunity reasoning
        if relative_value >= 1.0:
            bullets.append("Optimal timing - this is his best remaining opportunity")
        elif relative_value >= 0.85:
            bullets.append(f"Good timing - close to optimal value ({relative_value:.0%} of best)")
        elif relative_value >= 0.6:
            bullets.append(f"Consider saving for {best_future_event} ({relative_value:.0%} of best value)")
        else:
            bullets.append(f"Save for {best_future_event} - much better opportunity ahead")

        # 3. Win probability reasoning
        if sim.win_rate >= 0.10:
            if tournament.is_major or tournament.is_signature:
                bullets.append(f"Strong {sim.win_rate*100:.1f}% win probability at premium event")
            else:
                bullets.append(f"High {sim.win_rate*100:.1f}% win probability (consider saving for bigger event)")
        elif sim.win_rate >= 0.05:
            bullets.append(f"Solid {sim.win_rate*100:.1f}% win probability with {sim.top_10_rate*100:.0f}% top-10 chance")

        # 4. ENHANCED: Market consensus insight
        if market_explanation:
            bullets.append(market_explanation)

        # 5. ENHANCED: True skill insight (DG vs OWGR)
        if skill_explanation:
            bullets.append(skill_explanation)

        # 6. Field strength reasoning
        if field_strength == "WEAK":
            if golfer.owgr <= 50:
                bullets.append("Weak field gives mid-tier player strong winning chance")
        elif field_strength == "STRONG":
            if golfer.owgr <= 20:
                bullets.append("Strong field favors elite deployment now")

        # 7. Course history reasoning
        if course_history_summary and "win" in course_history_summary.lower():
            bullets.append(f"Proven winner here - {course_history_summary}")
        elif course_history_summary and "top-5" in course_history_summary.lower():
            bullets.append(f"Strong course history - {course_history_summary}")

        # 8. Cut probability reasoning
        if sim.cut_rate >= 0.90:
            bullets.append(f"Very safe pick - {sim.cut_rate*100:.0f}% make cut probability")
        elif sim.cut_rate < 0.75:
            bullets.append(f"Risky pick - only {sim.cut_rate*100:.0f}% make cut probability")

        # Limit to 5 bullets, prioritizing the first ones
        return bullets[:5]

    def _calculate_factor_contributions(
        self,
        base_ev: float,
        course_fit: float,
        timing_mult: float,
        field_mult: float,
        hedge_mult: float,
        phase_mult: float,
        course_history_boost: float,
        # ENHANCED: New API-based factors
        sg_match_mult: float = 1.0,
        market_mult: float = 1.0,
        skill_mult: float = 1.0,
    ) -> Dict[str, float]:
        """
        Calculate dollar contribution of each factor.
        Uses additive model with caps for transparency.
        """
        contributions = {}

        # Course fit: $15K per 0.1 SG/round, capped at ±$50K
        course_fit_adj = min(max(course_fit * 150000, -50000), 50000)
        contributions["course_fit"] = course_fit_adj

        # Timing (relative value): up to ±$50K
        timing_adj = min(max((timing_mult - 1.0) * base_ev, -50000), 50000)
        contributions["timing"] = timing_adj

        # Field strength: up to ±$30K
        field_adj = min(max((field_mult - 1.0) * base_ev, -30000), 30000)
        contributions["field_strength"] = field_adj

        # Hedge bonus: up to ±$15K
        hedge_adj = min(max((hedge_mult - 1.0) * base_ev, -15000), 15000)
        contributions["hedge"] = hedge_adj

        # Phase/elite deployment: up to ±$40K
        phase_adj = min(max((phase_mult - 1.0) * base_ev, -40000), 40000)
        contributions["phase"] = phase_adj

        # Course history: up to ±$25K
        history_adj = min(max((course_history_boost - 1.0) * base_ev, -25000), 25000)
        contributions["course_history"] = history_adj

        # ENHANCED: SG decomposition match: up to ±$35K
        sg_match_adj = min(max((sg_match_mult - 1.0) * base_ev, -35000), 35000)
        contributions["sg_match"] = sg_match_adj

        # ENHANCED: Market consensus edge: up to ±$20K
        market_adj = min(max((market_mult - 1.0) * base_ev, -20000), 20000)
        contributions["market_edge"] = market_adj

        # ENHANCED: True skill adjustment: up to ±$25K
        skill_adj = min(max((skill_mult - 1.0) * base_ev, -25000), 25000)
        contributions["true_skill"] = skill_adj

        return contributions

    def _calculate_timing_verdict(
        self,
        relative_value: float,
        best_future_event: str,
    ) -> str:
        """
        Calculate clear timing verdict for display.
        Returns: "USE NOW", "SAVE FOR [EVENT]", or "TOSS-UP"
        """
        if relative_value >= 0.95:
            return "USE NOW"
        elif relative_value >= 0.80:
            return "TOSS-UP"
        elif best_future_event:
            return f"SAVE FOR {best_future_event.upper()}"
        else:
            return "SAVE"

    def _calculate_confidence_pct(
        self,
        golfer: Golfer,
        sim: SimulationResult,
        course_history: 'CourseHistory',
        availability_status: str,
    ) -> int:
        """
        Calculate confidence percentage (0-100) based on data quality.
        """
        confidence = 50  # Base confidence

        # Simulation quality (+20 max)
        if sim.n_simulations >= 50000:
            confidence += 20
        elif sim.n_simulations >= 10000:
            confidence += 10

        # Course history data (+15 max)
        if course_history and course_history.years_played >= 4:
            confidence += 15
        elif course_history and course_history.years_played >= 2:
            confidence += 8

        # OWGR rank quality (+10 max)
        if golfer.owgr <= 20:
            confidence += 10
        elif golfer.owgr <= 50:
            confidence += 5

        # Availability status (+5 max)
        if availability_status == "confirmed":
            confidence += 5
        elif availability_status == "unlikely":
            confidence -= 10

        # Cut probability consistency
        if sim.cut_rate >= 0.85:
            confidence += 5

        return min(100, max(0, confidence))

    def _generate_risk_flags(
        self,
        golfer: Golfer,
        sim: SimulationResult,
        owgr_warning: bool,
        cut_warning: bool,
        availability_status: str,
    ) -> List[str]:
        """Generate list of risk flags for display."""
        flags = []

        if owgr_warning:
            flags.append(f"OWGR {golfer.owgr} (outside top 65)")

        if cut_warning:
            flags.append(f"Cut probability {sim.cut_rate*100:.0f}% (below 80%)")

        if availability_status == "unlikely":
            flags.append("May not play (UNLIKELY status)")

        if sim.std_earnings > sim.mean_earnings:
            flags.append("High variance (inconsistent)")

        return flags

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


    # =========================================================================
    # Phase 3.2: Multi-Entry Strategy Support
    # =========================================================================

    def get_multi_entry_recommendations(
        self,
        tournament: Tournament,
        num_entries: int = 2,
        top_n_per_entry: int = 3
    ) -> Dict[int, List[Recommendation]]:
        """
        Get differentiated recommendations for multiple entries.

        The strategy is to hedge by picking different golfers across entries:
        - Entry 1: Best EV picks (maximize expected value)
        - Entry 2+: Diversify with underowned/different golfers

        Returns dict of entry_id -> list of recommendations.
        """
        if num_entries < 1:
            return {}

        all_recs = self.get_recommendations(tournament, top_n=top_n_per_entry * num_entries * 2)

        if not all_recs:
            return {}

        result = {}
        used_golfers = set()

        for entry_num in range(1, num_entries + 1):
            entry_recs = []

            for rec in all_recs:
                if rec.golfer.name in used_golfers:
                    continue

                # For entry 1, just take top EV picks
                # For subsequent entries, prioritize different golfers
                if entry_num == 1:
                    entry_recs.append(rec)
                else:
                    # Boost hedge bonus importance for hedging entries
                    adjusted_rec = rec
                    entry_recs.append(adjusted_rec)

                if len(entry_recs) >= top_n_per_entry:
                    break

            # Mark golfers as used across entries
            for rec in entry_recs:
                used_golfers.add(rec.golfer.name)

            result[entry_num] = entry_recs

        return result

    def get_hedging_picks(
        self,
        tournament: Tournament,
        primary_pick: str,
        num_hedge_picks: int = 2
    ) -> List[Recommendation]:
        """
        Get hedge picks that are different from the primary pick.

        Good hedges are golfers with:
        - Different win probability tier
        - Low correlation with primary pick (different style)
        - Underused by opponents
        """
        all_recs = self.get_recommendations(tournament, top_n=20)

        if not all_recs:
            return []

        # Find the primary pick
        primary_rec = next((r for r in all_recs if r.golfer.name == primary_pick), None)

        if not primary_rec:
            # Primary not in top 20, just return top picks excluding primary
            return [r for r in all_recs if r.golfer.name != primary_pick][:num_hedge_picks]

        # Score potential hedges
        hedge_candidates = []
        primary_tier = self.classify_golfer_tier(primary_rec.golfer)
        primary_win_prob = primary_rec.golfer.win_probability

        for rec in all_recs:
            if rec.golfer.name == primary_pick:
                continue

            hedge_score = rec.expected_value

            # Bonus for different tier (diversification)
            rec_tier = self.classify_golfer_tier(rec.golfer)
            if rec_tier != primary_tier:
                hedge_score *= 1.1  # 10% bonus for tier diversification

            # Bonus for different win probability level
            rec_win_prob = rec.golfer.win_probability
            if abs(rec_win_prob - primary_win_prob) > 0.05:
                hedge_score *= 1.05  # 5% bonus for different win prob

            # Bonus for high hedge value (underused)
            if rec.hedge_bonus > 0.05:
                hedge_score *= 1.1

            hedge_candidates.append((rec, hedge_score))

        # Sort by hedge score
        hedge_candidates.sort(key=lambda x: x[1], reverse=True)

        return [r for r, _ in hedge_candidates[:num_hedge_picks]]

    # =========================================================================
    # Phase 3.3: Segment Optimization
    # =========================================================================

    def get_segment_optimization(
        self,
        num_segments: int = 6
    ) -> Dict:
        """
        Optimize picks within season segments.

        Divides the season into segments (e.g., 6 segments of ~6-7 events each)
        and ensures proper elite distribution across segments.

        Strategy:
        - Each segment should have 0-1 elite picks
        - Majors get priority for elites
        - Balance risk across segments
        """
        schedule = get_schedule()
        today = date.today()
        remaining = [t for t in schedule if t.date >= today]

        if not remaining:
            return {"error": "No remaining tournaments"}

        # Divide into segments
        segment_size = max(1, len(remaining) // num_segments)
        segments = []

        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size if i < num_segments - 1 else len(remaining)
            segment_tournaments = remaining[start_idx:end_idx]

            if segment_tournaments:
                segments.append({
                    "segment_num": i + 1,
                    "tournaments": segment_tournaments,
                    "has_major": any(t.is_major for t in segment_tournaments),
                    "has_signature": any(t.is_signature for t in segment_tournaments),
                    "total_purse": sum(t.purse for t in segment_tournaments),
                    "recommended_elite_count": 1 if any(t.is_major for t in segment_tournaments) else 0,
                })

        # Get available golfers
        used_golfers = set(self.db.get_used_golfers())
        all_golfers = self.db.get_all_golfers()
        available_elites = [g for g in all_golfers if g.owgr <= 20 and g.name not in used_golfers]
        available_mid = [g for g in all_golfers if 20 < g.owgr <= 50 and g.name not in used_golfers]

        # Allocate elites to segments with majors
        elite_allocations = {}
        remaining_elites = list(available_elites)

        for segment in segments:
            if segment["has_major"] and remaining_elites:
                # Find the major in this segment
                major = next((t for t in segment["tournaments"] if t.is_major), None)
                if major:
                    # Assign best available elite
                    best_elite = min(remaining_elites, key=lambda g: g.owgr)
                    elite_allocations[major.name] = best_elite.name
                    remaining_elites.remove(best_elite)
                    segment["assigned_elite"] = best_elite.name
                    segment["elite_tournament"] = major.name

        # Build segment recommendations
        segment_recommendations = []
        for segment in segments:
            segment_recs = {
                "segment_num": segment["segment_num"],
                "start_date": segment["tournaments"][0].date.strftime("%b %d"),
                "end_date": segment["tournaments"][-1].date.strftime("%b %d"),
                "num_events": len(segment["tournaments"]),
                "has_major": segment["has_major"],
                "has_signature": segment["has_signature"],
                "total_purse": segment["total_purse"],
                "elite_pick": segment.get("assigned_elite"),
                "elite_tournament": segment.get("elite_tournament"),
                "tournament_names": [t.name for t in segment["tournaments"]],
            }
            segment_recommendations.append(segment_recs)

        return {
            "num_segments": num_segments,
            "total_remaining_events": len(remaining),
            "available_elites": len(available_elites),
            "available_mid_tier": len(available_mid),
            "segments": segment_recommendations,
            "elite_allocations": elite_allocations,
            "strategy_notes": self._get_segment_strategy_notes(segments),
        }

    def _get_segment_strategy_notes(self, segments: List[Dict]) -> List[str]:
        """Generate strategy notes for segment optimization."""
        notes = []

        # Count majors and signatures
        majors = sum(1 for s in segments if s["has_major"])
        signatures = sum(1 for s in segments if s["has_signature"])

        notes.append(f"Season has {majors} majors and {signatures} signature events remaining")

        # Elite deployment advice
        elites_assigned = sum(1 for s in segments if s.get("assigned_elite"))
        notes.append(f"Recommended elite deployments: {elites_assigned} (for majors)")

        # Risk distribution
        high_value_segments = sum(1 for s in segments if s["total_purse"] > 50_000_000)
        notes.append(f"High-value segments (>$50M purse): {high_value_segments}")

        return notes


def get_strategy() -> Strategy:
    """Get configured strategy engine."""
    return Strategy()
