"""
Monte Carlo simulation engine for PGA One and Done Optimizer.
Runs probabilistic simulations to estimate expected value and variance.
"""

import logging
from typing import List, Optional, Dict, Tuple
from multiprocessing import Pool, cpu_count
from datetime import datetime

import numpy as np

from .config import get_config, PAYOUT_DISTRIBUTION
from .database import Database
from .models import (
    Tournament, Golfer, SimulationResult, WhatIfScenario,
    Recommendation, SeasonPlan, Tier
)

logger = logging.getLogger(__name__)


class Simulator:
    """Monte Carlo simulation engine."""

    def __init__(self, n_simulations: int = None):
        """Initialize simulator."""
        config = get_config()
        self.n_simulations = n_simulations or config.default_simulations
        self.db = Database()
        self._rng = np.random.default_rng()

    def _generate_finish_position(
        self,
        win_prob: float,
        top_10_prob: float,
        top_20_prob: float,
        make_cut_prob: float,
        field_size: int = 156
    ) -> int:
        """
        Generate a random finish position based on probabilities.
        Returns position (1-based) or 0 for missed cut.
        """
        roll = self._rng.random()

        # Win
        if roll < win_prob:
            return 1

        # 2nd-5th (top 5 but not win)
        top_5_prob = (top_10_prob * 0.5)  # Approximate half of top-10 is top-5
        if roll < top_5_prob:
            return self._rng.integers(2, 6)

        # 6th-10th
        if roll < top_10_prob:
            return self._rng.integers(6, 11)

        # 11th-20th
        if roll < top_20_prob:
            return self._rng.integers(11, 21)

        # Made cut but worse than 20th
        if roll < make_cut_prob:
            return self._rng.integers(21, 71)  # Typical cut line around 70

        # Missed cut
        return 0

    def _calculate_payout(self, position: int, purse: int) -> int:
        """Calculate payout for a finish position."""
        if position == 0:
            return 0  # Missed cut

        if position in PAYOUT_DISTRIBUTION:
            return int(purse * PAYOUT_DISTRIBUTION[position])
        elif position <= 70:
            # Extrapolate for positions not in table
            return int(purse * 0.004)  # ~$40K on $10M purse
        return 0

    def simulate_tournament(
        self,
        golfer: Golfer,
        tournament: Tournament,
        n_simulations: int = None
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation for a golfer in a tournament.
        Returns detailed statistics about expected performance.
        """
        n = n_simulations or self.n_simulations

        # Check cache
        cached = self.db.get_simulation(golfer.name, tournament.name)
        if cached and cached.n_simulations >= n:
            return cached

        logger.info(f"Running {n:,} simulations for {golfer.name} at {tournament.name}")

        # Get probabilities
        probs = self.db.get_golfer_probability(golfer.name, tournament.name)
        if probs:
            win_prob = probs.get("win_prob", golfer.win_probability)
            top_10_prob = probs.get("top_10_prob", golfer.top_10_probability)
            top_20_prob = probs.get("top_20_prob", golfer.top_20_probability)
            make_cut_prob = probs.get("make_cut_prob", golfer.make_cut_probability)
        else:
            # Use golfer's general probabilities or defaults based on OWGR
            win_prob = golfer.win_probability or self._owgr_to_win_prob(golfer.owgr)
            top_10_prob = golfer.top_10_probability or self._owgr_to_top10_prob(golfer.owgr)
            top_20_prob = golfer.top_20_probability or top_10_prob * 1.8
            make_cut_prob = golfer.make_cut_probability or self._owgr_to_cut_prob(golfer.owgr)

        # Run simulations
        earnings = np.zeros(n)
        wins = 0
        top_10s = 0
        cuts_made = 0

        for i in range(n):
            position = self._generate_finish_position(
                win_prob, top_10_prob, top_20_prob, make_cut_prob
            )
            payout = self._calculate_payout(position, tournament.purse)
            earnings[i] = payout

            if position == 1:
                wins += 1
            if 0 < position <= 10:
                top_10s += 1
            if position > 0:
                cuts_made += 1

        result = SimulationResult(
            golfer_name=golfer.name,
            tournament_name=tournament.name,
            n_simulations=n,
            mean_earnings=float(np.mean(earnings)),
            median_earnings=float(np.median(earnings)),
            std_earnings=float(np.std(earnings)),
            percentile_10=float(np.percentile(earnings, 10)),
            percentile_25=float(np.percentile(earnings, 25)),
            percentile_75=float(np.percentile(earnings, 75)),
            percentile_90=float(np.percentile(earnings, 90)),
            win_count=wins,
            top_10_count=top_10s,
            cut_made_count=cuts_made,
        )

        # Cache result
        self.db.save_simulation(result)

        logger.info(
            f"Simulation complete: EV=${result.mean_earnings:,.0f}, "
            f"Win%={result.win_rate*100:.2f}%, Top10%={result.top_10_rate*100:.1f}%"
        )

        return result

    def _owgr_to_win_prob(self, owgr: int) -> float:
        """Estimate win probability from OWGR."""
        if owgr <= 5:
            return 0.12
        elif owgr <= 10:
            return 0.08
        elif owgr <= 20:
            return 0.05
        elif owgr <= 50:
            return 0.02
        elif owgr <= 100:
            return 0.008
        else:
            return 0.002

    def _owgr_to_top10_prob(self, owgr: int) -> float:
        """Estimate top-10 probability from OWGR."""
        if owgr <= 5:
            return 0.45
        elif owgr <= 10:
            return 0.35
        elif owgr <= 20:
            return 0.28
        elif owgr <= 50:
            return 0.18
        elif owgr <= 100:
            return 0.10
        else:
            return 0.05

    def _owgr_to_cut_prob(self, owgr: int) -> float:
        """Estimate make-cut probability from OWGR."""
        if owgr <= 10:
            return 0.92
        elif owgr <= 25:
            return 0.85
        elif owgr <= 50:
            return 0.78
        elif owgr <= 100:
            return 0.68
        else:
            return 0.55

    def calculate_ev(self, golfer: Golfer, tournament: Tournament) -> float:
        """Calculate expected value for a golfer-tournament combination."""
        result = self.simulate_tournament(golfer, tournament)
        return result.mean_earnings

    def simulate_season(
        self,
        picks: List[Tuple[Golfer, Tournament]],
        n_simulations: int = None
    ) -> Dict:
        """
        Simulate full season with given picks.
        Returns season-level statistics.
        """
        n = n_simulations or min(self.n_simulations, 10000)  # Reduced for season

        logger.info(f"Running {n:,} season simulations for {len(picks)} picks")

        season_earnings = np.zeros(n)
        season_cuts = np.zeros(n)
        season_top10s = np.zeros(n)
        season_wins = np.zeros(n)

        for golfer, tournament in picks:
            # Get probabilities
            probs = self.db.get_golfer_probability(golfer.name, tournament.name)
            win_prob = probs.get("win_prob", 0.02) if probs else 0.02
            top_10_prob = probs.get("top_10_prob", 0.15) if probs else 0.15
            top_20_prob = probs.get("top_20_prob", 0.25) if probs else 0.25
            make_cut_prob = probs.get("make_cut_prob", 0.70) if probs else 0.70

            for i in range(n):
                position = self._generate_finish_position(
                    win_prob, top_10_prob, top_20_prob, make_cut_prob
                )
                earnings = self._calculate_payout(position, tournament.purse)
                season_earnings[i] += earnings

                if position > 0:
                    season_cuts[i] += 1
                if 0 < position <= 10:
                    season_top10s[i] += 1
                if position == 1:
                    season_wins[i] += 1

        return {
            "total_picks": len(picks),
            "mean_earnings": float(np.mean(season_earnings)),
            "median_earnings": float(np.median(season_earnings)),
            "std_earnings": float(np.std(season_earnings)),
            "percentile_10": float(np.percentile(season_earnings, 10)),
            "percentile_90": float(np.percentile(season_earnings, 90)),
            "mean_cuts": float(np.mean(season_cuts)),
            "mean_top10s": float(np.mean(season_top10s)),
            "mean_wins": float(np.mean(season_wins)),
        }

    def regret_analysis(
        self,
        selected_golfer: Golfer,
        tournament: Tournament,
        alternatives: List[Golfer],
        n_simulations: int = None
    ) -> Dict:
        """
        Analyze potential regret of selecting one golfer over alternatives.
        """
        n = n_simulations or min(self.n_simulations, 10000)

        # Simulate selected golfer
        selected_result = self.simulate_tournament(selected_golfer, tournament, n)

        # Simulate alternatives
        alt_results = []
        for alt in alternatives[:5]:  # Limit to top 5 alternatives
            alt_result = self.simulate_tournament(alt, tournament, n)
            alt_results.append(alt_result)

        if not alt_results:
            return {
                "regret_risk": 0,
                "upside_vs_field": selected_result.mean_earnings,
                "best_alternative": None,
            }

        # Find best alternative
        best_alt = max(alt_results, key=lambda r: r.mean_earnings)

        # Calculate regret (how much we lose if alternative wins)
        regret = max(0, best_alt.mean_earnings - selected_result.mean_earnings)

        # Calculate upside (how much we gain vs average alternative)
        avg_alt_ev = np.mean([r.mean_earnings for r in alt_results])
        upside = selected_result.mean_earnings - avg_alt_ev

        return {
            "regret_risk": regret,
            "upside_vs_field": upside,
            "selected_ev": selected_result.mean_earnings,
            "best_alternative": best_alt.golfer_name,
            "best_alternative_ev": best_alt.mean_earnings,
            "avg_alternative_ev": avg_alt_ev,
        }

    def what_if_analysis(
        self,
        golfer: Golfer,
        tournament: Tournament,
        alternative: Golfer,
    ) -> WhatIfScenario:
        """
        Compare two golfers for the same tournament.
        """
        main_result = self.simulate_tournament(golfer, tournament)
        alt_result = self.simulate_tournament(alternative, tournament)

        return WhatIfScenario(
            scenario_description=f"Pick {golfer.name} vs {alternative.name} for {tournament.name}",
            golfer_name=golfer.name,
            tournament_name=tournament.name,
            expected_outcome=main_result,
            alternative_golfer=alternative.name,
            alternative_outcome=alt_result,
            regret_if_wrong=max(0, alt_result.mean_earnings - main_result.mean_earnings),
            upside_if_right=max(0, main_result.mean_earnings - alt_result.mean_earnings),
        )

    def simulate_remaining_season(
        self,
        available_golfers: List[Golfer],
        remaining_tournaments: List[Tournament],
        strategy: str = "ev_max"
    ) -> SeasonPlan:
        """
        Simulate and optimize picks for remaining season.
        strategy: 'ev_max' (maximize EV), 'conservative' (consistent), 'aggressive' (high variance)
        """
        plan = SeasonPlan()
        used = set()

        for tournament in remaining_tournaments:
            best_golfer = None
            best_score = -1

            for golfer in available_golfers:
                if golfer.name in used:
                    continue

                result = self.simulate_tournament(golfer, tournament)

                if strategy == "ev_max":
                    score = result.mean_earnings
                elif strategy == "conservative":
                    # Prefer consistent performers (lower std, higher cut rate)
                    score = result.mean_earnings * result.cut_rate
                elif strategy == "aggressive":
                    # Prefer high upside (90th percentile)
                    score = result.percentile_90
                else:
                    score = result.mean_earnings

                if score > best_score:
                    best_score = score
                    best_golfer = golfer

            if best_golfer:
                plan.planned_picks.append({
                    "tournament": tournament.name,
                    "golfer": best_golfer.name,
                    "ev": best_score,
                })
                used.add(best_golfer.name)
                plan.projected_earnings += int(best_score)

        plan.used_golfers = list(used)
        return plan

    def batch_simulate(
        self,
        golfers: List[Golfer],
        tournament: Tournament,
        n_simulations: int = None
    ) -> List[SimulationResult]:
        """
        Run simulations for multiple golfers in parallel.
        """
        n = n_simulations or self.n_simulations

        # Use multiprocessing for large batches
        if len(golfers) > 10:
            with Pool(processes=min(cpu_count(), 4)) as pool:
                args = [(g, tournament, n // 10) for g in golfers]  # Reduced sims for batch
                results = pool.starmap(self._simulate_single, args)
        else:
            results = [
                self.simulate_tournament(g, tournament, n)
                for g in golfers
            ]

        return sorted(results, key=lambda r: r.mean_earnings, reverse=True)

    def _simulate_single(
        self,
        golfer: Golfer,
        tournament: Tournament,
        n: int
    ) -> SimulationResult:
        """Helper for parallel simulation."""
        return self.simulate_tournament(golfer, tournament, n)


def get_simulator(n_simulations: int = None) -> Simulator:
    """Get configured simulator instance."""
    return Simulator(n_simulations)
