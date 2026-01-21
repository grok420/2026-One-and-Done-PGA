"""
Data models for PGA One and Done Optimizer.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional, List, Dict
from enum import Enum


class Tier(Enum):
    """Tournament tier based on purse size."""
    TIER_1 = 1  # $20M+
    TIER_2 = 2  # $9M - $15M
    TIER_3 = 3  # < $9M


class CutRule(Enum):
    """Tournament cut rules for 2026."""
    STANDARD = "standard"  # 65 and ties (typical PGA event)
    TOP_50_TIES = "top_50"  # Top 50 + ties + within 10 shots (signature events)
    NO_CUT = "no_cut"  # No cut (5 signature events + Tour Championship)


class SeasonPhase(Enum):
    """Season phase for strategy."""
    EARLY = "early"      # Jan-Mar: Build with mid-tiers
    MID = "mid"          # Apr-Jul: Deploy elites for majors
    PLAYOFF = "playoff"  # Aug: FedEx events


@dataclass
class Tournament:
    """Represents a PGA Tour tournament."""
    name: str
    date: date
    purse: int
    course: str = ""
    tier: Tier = Tier.TIER_2
    is_major: bool = False
    is_signature: bool = False
    is_playoff: bool = False
    is_opposite_field: bool = False
    cut_rule: CutRule = CutRule.STANDARD
    field_size: int = 144  # Updated from 156 for 2026

    @property
    def has_cut(self) -> bool:
        """Whether the tournament has a cut."""
        return self.cut_rule != CutRule.NO_CUT

    @property
    def winner_share(self) -> int:
        """Winner typically gets 18% of purse."""
        return int(self.purse * 0.18)

    @property
    def min_payout(self) -> int:
        """Minimum payout for no-cut events (last place gets paid)."""
        if not self.has_cut:
            # Last place in no-cut events gets approximately 0.3% of purse
            return int(self.purse * 0.003)
        return 0

    def get_payout(self, position: int) -> int:
        """Estimate payout for finishing position."""
        # Standard PGA payout percentages (approximate)
        payouts = {
            1: 0.180, 2: 0.109, 3: 0.069, 4: 0.049, 5: 0.041,
            6: 0.036, 7: 0.0335, 8: 0.031, 9: 0.029, 10: 0.027,
            11: 0.025, 12: 0.023, 13: 0.0215, 14: 0.020, 15: 0.0195,
            16: 0.019, 17: 0.0185, 18: 0.018, 19: 0.0175, 20: 0.017,
            21: 0.0165, 22: 0.016, 23: 0.0155, 24: 0.015, 25: 0.0145,
            26: 0.014, 27: 0.0135, 28: 0.013, 29: 0.0125, 30: 0.012,
        }
        if position in payouts:
            return int(self.purse * payouts[position])
        elif position <= 50:
            return int(self.purse * 0.005)  # Roughly for 31-50
        elif position <= 70:
            return int(self.purse * 0.003)  # Roughly for 51-70
        return 0  # Missed cut or worse


@dataclass
class ApproachBuckets:
    """Approach shot performance by yardage bucket (strokes gained)."""
    sg_50_100: float = 0.0    # 50-100 yards
    sg_100_150: float = 0.0   # 100-150 yards
    sg_150_200: float = 0.0   # 150-200 yards
    sg_200_plus: float = 0.0  # 200+ yards
    # Lie-based adjustments
    sg_fairway: float = 0.0   # Approach from fairway
    sg_rough: float = 0.0     # Approach from rough


@dataclass
class GolferStats:
    """Strokes gained and other statistics."""
    sg_total: float = 0.0
    sg_off_tee: float = 0.0
    sg_approach: float = 0.0
    sg_around_green: float = 0.0
    sg_putting: float = 0.0
    driving_distance: float = 0.0
    driving_accuracy: float = 0.0
    gir_pct: float = 0.0
    scrambling_pct: float = 0.0
    approach_buckets: Optional[ApproachBuckets] = None


@dataclass
class Golfer:
    """Represents a PGA Tour golfer."""
    name: str
    owgr: int = 999  # Official World Golf Ranking
    datagolf_id: Optional[str] = None
    stats: GolferStats = field(default_factory=GolferStats)
    course_history: Dict[str, float] = field(default_factory=dict)  # course -> avg finish
    win_probability: float = 0.0
    top_10_probability: float = 0.0
    top_20_probability: float = 0.0
    make_cut_probability: float = 0.0
    # Course fit adjustment from Data Golf (SG/round adjustment based on course fit)
    course_fit_adjustment: float = 0.0

    @property
    def is_elite(self) -> bool:
        """Top-20 OWGR considered elite."""
        return self.owgr <= 20

    @property
    def is_mid_tier(self) -> bool:
        """21-50 OWGR considered mid-tier."""
        return 20 < self.owgr <= 50

    @property
    def is_risky_owgr(self) -> bool:
        """OWGR 66+ is risky - last year's winner never picked anyone outside top 65."""
        return self.owgr > 65


@dataclass
class Pick:
    """A golfer pick for a tournament."""
    golfer_name: str
    tournament_name: str
    tournament_date: date
    earnings: int = 0
    position: Optional[int] = None
    made_cut: bool = False
    is_major: bool = False


@dataclass
class LeagueStanding:
    """A player's standing in the league."""
    rank: int
    player_name: str
    username: str
    total_earnings: int
    cuts_made: int = 0
    picks_made: int = 0
    majors_earnings: int = 0  # For side pool


@dataclass
class OpponentPick:
    """Track what golfers opponents have used."""
    opponent_username: str
    golfer_name: str
    tournament_name: str
    tournament_date: date


@dataclass
class Recommendation:
    """A recommended pick with analysis."""
    golfer: Golfer
    tournament: Tournament
    expected_value: float
    win_ev: float = 0.0
    top_10_ev: float = 0.0
    cut_ev: float = 0.0
    confidence: float = 0.0  # 0-1 confidence score
    hedge_bonus: float = 0.0  # Differentiation from opponents
    regret_risk: float = 0.0  # Opportunity cost if wrong
    reasoning: str = ""
    course_fit_sg: float = 0.0  # Course fit adjustment (SG/round)
    owgr_warning: bool = False  # True if golfer OWGR > 65

    @property
    def total_score(self) -> float:
        """Combined score for ranking."""
        return self.expected_value + self.hedge_bonus - (self.regret_risk * 0.1)

    @property
    def has_owgr_risk(self) -> bool:
        """Check if pick has OWGR risk (>65 never won in prior year)."""
        return self.golfer.owgr > 65


@dataclass
class SimulationResult:
    """Results from Monte Carlo simulation."""
    golfer_name: str
    tournament_name: str
    n_simulations: int
    mean_earnings: float
    median_earnings: float
    std_earnings: float
    percentile_10: float
    percentile_25: float
    percentile_75: float
    percentile_90: float
    win_count: int
    top_10_count: int
    cut_made_count: int

    @property
    def win_rate(self) -> float:
        return self.win_count / self.n_simulations if self.n_simulations > 0 else 0

    @property
    def top_10_rate(self) -> float:
        return self.top_10_count / self.n_simulations if self.n_simulations > 0 else 0

    @property
    def cut_rate(self) -> float:
        return self.cut_made_count / self.n_simulations if self.n_simulations > 0 else 0


@dataclass
class SeasonPlan:
    """Planned picks for the season."""
    planned_picks: List[Dict[str, str]] = field(default_factory=list)  # [{tournament, golfer}]
    elite_golfers_reserved: List[str] = field(default_factory=list)  # For majors
    mid_tier_golfers_available: List[str] = field(default_factory=list)
    used_golfers: List[str] = field(default_factory=list)
    projected_earnings: int = 0
    projected_rank: int = 0


@dataclass
class WhatIfScenario:
    """What-if analysis result."""
    scenario_description: str
    golfer_name: str
    tournament_name: str
    expected_outcome: SimulationResult
    alternative_golfer: str
    alternative_outcome: SimulationResult
    regret_if_wrong: float  # Potential loss vs alternative
    upside_if_right: float  # Potential gain vs alternative
