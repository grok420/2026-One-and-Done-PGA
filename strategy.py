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
        SimulationResult, LeagueStanding
    )
    from .simulator import Simulator
    from .api import DataGolfAPI
except ImportError:
    from config import get_config, get_schedule, get_next_tournament, get_majors, get_no_cut_events, get_course_profile, CourseProfile
    from database import Database
    from models import (
        Tournament, Golfer, Recommendation, SeasonPhase, Tier, CutRule,
        SimulationResult, LeagueStanding
    )
    from simulator import Simulator
    from api import DataGolfAPI

# OWGR threshold for warnings - last year's winner never picked outside top 65
OWGR_WARNING_THRESHOLD = 65

# Tournament value thresholds (purse in millions)
HIGH_VALUE_PURSE = 15_000_000  # Majors, Players, signatures
MID_VALUE_PURSE = 10_000_000   # Strong regular events
BASE_PURSE = 8_000_000         # Standard events

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

    def check_owgr_warning(self, golfer: Golfer) -> Tuple[bool, str]:
        """
        Check if golfer has OWGR risk.
        Last year's One and Done winner never picked anyone outside top 65 OWGR.
        Returns (has_warning, warning_message).
        """
        if golfer.owgr > OWGR_WARNING_THRESHOLD:
            return True, f"OWGR {golfer.owgr} is outside top {OWGR_WARNING_THRESHOLD} - historical winners avoid these picks"
        return False, ""

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
        """
        base_factor = 1.0

        # Purse-based value (bigger purse = more valuable)
        purse_factor = tournament.purse / BASE_PURSE

        # Tournament type bonuses
        if tournament.is_major:
            # Majors are THE events to deploy elites - winner gets $3.6-4.5M
            type_bonus = 2.0
        elif tournament.is_signature:
            # Signature events have $20M purses, winners get $3.6M
            type_bonus = 1.75
        elif tournament.is_playoff:
            # FedEx Playoffs - Tour Championship is huge
            type_bonus = 1.6
        elif tournament.tier == Tier.TIER_1:
            type_bonus = 1.3
        else:
            type_bonus = 1.0

        return base_factor * purse_factor * type_bonus

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

        # Get standings-based strategy adjustment
        standings_mode, standings_risk_mult, standings_explanation = self.get_standings_strategy()
        logger.info(f"Standings strategy: {standings_mode} ({standings_explanation})")

        recommendations = []

        # Calculate tournament value for context
        tournament_value = self.calculate_tournament_value_factor(tournament)
        is_high_value_event = tournament.is_major or tournament.is_signature or tournament.purse >= HIGH_VALUE_PURSE

        for golfer in golfers:
            # Check for LIV golfer at non-major (skip with warning)
            liv_warning, liv_msg = self.check_liv_warning(golfer, tournament)
            if liv_warning:
                logger.info(f"Skipping {golfer.name}: {liv_msg}")
                continue  # Don't recommend LIV golfers for non-majors
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

            # Final score combines:
            # - Base EV (simulation mean)
            # - Win probability value (don't waste high win% at small events)
            # - Hedge bonus (differentiation from opponents)
            # - Phase multiplier (save elites early, deploy late)
            # - Elite deployment factor
            # - Course fit adjustment
            # - Standings-based risk adjustment
            expected_value = base_ev * win_prob_multiplier * hedge_bonus * phase_multiplier * elite_save_penalty * course_fit_ev_factor * standings_adjustment

            # Check OWGR warning
            owgr_warning, owgr_msg = self.check_owgr_warning(golfer)

            # Build reasoning (include OWGR warning and elite save advice if applicable)
            reasoning = self._build_reasoning(
                golfer, tournament, sim_result, hedge_bonus, phase, regret_risk
            )
            if owgr_warning:
                reasoning = f"WARNING: {owgr_msg} | {reasoning}"

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

            # Add standings context if adjusting strategy
            if standings_adjustment != 1.0:
                if standings_mode == "protect":
                    reasoning = f"STANDINGS: Protecting lead - favor consistency | {reasoning}"
                elif standings_mode in ("aggressive", "desperation"):
                    reasoning = f"STANDINGS: Trailing - need upside ({standings_adjustment:.0%} boost) | {reasoning}"

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
