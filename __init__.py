"""
PGA One and Done Optimizer
A comprehensive fantasy golf optimization tool.
"""

__version__ = "1.0.0"
__author__ = "Eric"

from .models import (
    Tournament, Golfer, GolferStats, Pick, LeagueStanding,
    OpponentPick, Recommendation, SimulationResult, Tier, SeasonPhase
)
from .config import get_config, get_schedule, get_next_tournament
from .database import Database
from .api import DataGolfAPI, get_api
from .scraper import Scraper, get_scraper
from .simulator import Simulator, get_simulator
from .strategy import Strategy, get_strategy

__all__ = [
    # Models
    "Tournament", "Golfer", "GolferStats", "Pick", "LeagueStanding",
    "OpponentPick", "Recommendation", "SimulationResult", "Tier", "SeasonPhase",
    # Config
    "get_config", "get_schedule", "get_next_tournament",
    # Core classes
    "Database", "DataGolfAPI", "Scraper", "Simulator", "Strategy",
    # Factory functions
    "get_api", "get_scraper", "get_simulator", "get_strategy",
]
