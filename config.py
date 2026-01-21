"""
Configuration management for PGA One and Done Optimizer.
Includes 2026 PGA Tour schedule with official purses.
"""

import os
from pathlib import Path
from datetime import date
from typing import Dict, List, Optional
from dataclasses import dataclass

from dotenv import load_dotenv

from .models import Tournament, Tier, CutRule


# Load environment variables from .env file
load_dotenv()


# Field size constants (updated for 2026)
DEFAULT_FIELD_SIZE = 144  # Down from 156
SIGNATURE_FIELD_SIZE = 70  # Signature events have smaller fields
NO_CUT_FIELD_SIZE = 70  # No-cut events


@dataclass
class Config:
    """Application configuration."""
    # User credentials (from environment)
    site_email: str = ""
    site_password: str = ""
    site_username: str = ""
    datagolf_api_key: str = ""

    # League settings
    league_name: str = "Bushwood"
    league_size_min: int = 75
    league_size_max: int = 90

    # Strategy settings
    risk_level: int = 5  # 1-10, 10 being most aggressive
    elite_reserve_count: int = 4  # Elites to save for majors

    # Paths
    data_dir: Path = Path.home() / ".pga_one_and_done"
    db_path: Path = Path.home() / ".pga_one_and_done" / "data.db"
    cache_dir: Path = Path.home() / ".pga_one_and_done" / "cache"

    # Scraping settings
    max_requests_per_session: int = 10
    cache_expiry_hours: int = 24
    request_delay_seconds: float = 2.0
    max_retries: int = 5

    # Simulation settings
    default_simulations: int = 50000

    # URLs
    site_base_url: str = "https://www.buzzfantasygolf.com"
    datagolf_base_url: str = "https://feeds.datagolf.com"

    def __post_init__(self):
        """Load credentials from environment."""
        self.site_email = os.getenv("PGA_OAD_EMAIL", "gitberge@gmail.com")
        self.site_password = os.getenv("PGA_OAD_PASSWORD", "Sixers123!")
        self.site_username = os.getenv("PGA_OAD_USERNAME", "gitberge")
        self.datagolf_api_key = os.getenv("DATAGOLF_API_KEY", "")

        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def save_to_env(self, env_path: Optional[Path] = None):
        """Save credentials to .env file."""
        env_path = env_path or (self.data_dir / ".env")
        with open(env_path, "w") as f:
            f.write(f"PGA_OAD_EMAIL={self.site_email}\n")
            f.write(f"PGA_OAD_PASSWORD={self.site_password}\n")
            f.write(f"PGA_OAD_USERNAME={self.site_username}\n")
            f.write(f"DATAGOLF_API_KEY={self.datagolf_api_key}\n")
            f.write(f"PGA_OAD_RISK_LEVEL={self.risk_level}\n")


def get_config() -> Config:
    """Get application configuration."""
    return Config()


# ============================================================================
# 2026 PGA TOUR SCHEDULE - HARDCODED FALLBACK
# Official purses from PGA Tour
# ============================================================================

SCHEDULE_2026: List[Tournament] = [
    # January
    Tournament(
        name="Sony Open in Hawaii",
        date=date(2026, 1, 15),
        purse=9_100_000,
        course="Waialae Country Club",
        tier=Tier.TIER_2,
    ),
    Tournament(
        name="The American Express",
        date=date(2026, 1, 22),
        purse=9_200_000,
        course="PGA West",
        tier=Tier.TIER_2,
    ),
    Tournament(
        name="Farmers Insurance Open",
        date=date(2026, 1, 29),
        purse=9_600_000,
        course="Torrey Pines",
        tier=Tier.TIER_2,
    ),

    # February
    Tournament(
        name="Waste Management Phoenix Open",
        date=date(2026, 2, 5),
        purse=9_600_000,
        course="TPC Scottsdale",
        tier=Tier.TIER_2,
    ),
    Tournament(
        name="AT&T Pebble Beach Pro-Am",
        date=date(2026, 2, 12),
        purse=20_000_000,
        course="Pebble Beach Golf Links",
        tier=Tier.TIER_1,
        is_signature=True,
        cut_rule=CutRule.NO_CUT,  # No-cut signature event
        field_size=SIGNATURE_FIELD_SIZE,
    ),
    Tournament(
        name="Genesis Invitational",
        date=date(2026, 2, 19),
        purse=20_000_000,
        course="Riviera Country Club",
        tier=Tier.TIER_1,
        is_signature=True,
        cut_rule=CutRule.TOP_50_TIES,  # 36-hole cut to top 50+ties
        field_size=SIGNATURE_FIELD_SIZE,
    ),
    Tournament(
        name="Cognizant Classic",
        date=date(2026, 2, 26),
        purse=9_600_000,
        course="PGA National",
        tier=Tier.TIER_2,
    ),

    # March
    Tournament(
        name="Arnold Palmer Invitational",
        date=date(2026, 3, 5),
        purse=20_000_000,
        course="Bay Hill Club",
        tier=Tier.TIER_1,
        is_signature=True,
        cut_rule=CutRule.TOP_50_TIES,  # 36-hole cut to top 50+ties
        field_size=SIGNATURE_FIELD_SIZE,
    ),
    Tournament(
        name="Puerto Rico Open",
        date=date(2026, 3, 5),
        purse=4_000_000,
        course="Grand Reserve Golf Club",
        tier=Tier.TIER_3,
        is_opposite_field=True,
    ),
    Tournament(
        name="The Players Championship",
        date=date(2026, 3, 12),
        purse=25_000_000,
        course="TPC Sawgrass",
        tier=Tier.TIER_1,
        is_signature=True,
        field_size=144,  # The Players has larger field
    ),
    Tournament(
        name="Valspar Championship",
        date=date(2026, 3, 19),
        purse=9_100_000,
        course="Innisbrook Resort",
        tier=Tier.TIER_2,
    ),
    Tournament(
        name="Texas Children's Houston Open",
        date=date(2026, 3, 26),
        purse=9_900_000,
        course="Memorial Park Golf Course",
        tier=Tier.TIER_2,
    ),

    # April
    Tournament(
        name="Valero Texas Open",
        date=date(2026, 4, 2),
        purse=9_800_000,
        course="TPC San Antonio",
        tier=Tier.TIER_2,
    ),
    Tournament(
        name="The Masters",
        date=date(2026, 4, 9),
        purse=20_000_000,
        course="Augusta National",
        tier=Tier.TIER_1,
        is_major=True,
    ),
    Tournament(
        name="RBC Heritage",
        date=date(2026, 4, 16),
        purse=20_000_000,
        course="Harbour Town Golf Links",
        tier=Tier.TIER_1,
        is_signature=True,
        cut_rule=CutRule.NO_CUT,  # No-cut signature event
        field_size=SIGNATURE_FIELD_SIZE,
    ),
    Tournament(
        name="Zurich Classic of New Orleans",
        date=date(2026, 4, 23),
        purse=9_500_000,
        course="TPC Louisiana",
        tier=Tier.TIER_2,
    ),
    # May
    Tournament(
        name="Miami Championship",
        date=date(2026, 5, 1),
        purse=20_000_000,
        course="Trump National Doral",
        tier=Tier.TIER_1,
        is_signature=True,
        cut_rule=CutRule.NO_CUT,  # No-cut signature event (NEW for 2026)
        field_size=SIGNATURE_FIELD_SIZE,
    ),
    Tournament(
        name="Truist Championship",
        date=date(2026, 5, 7),
        purse=20_000_000,
        course="Quail Hollow Club",
        tier=Tier.TIER_1,
        is_signature=True,
        cut_rule=CutRule.NO_CUT,  # No-cut signature event
        field_size=SIGNATURE_FIELD_SIZE,
    ),
    Tournament(
        name="Myrtle Beach Classic",
        date=date(2026, 5, 7),
        purse=4_000_000,
        course="Dunes Golf and Beach Club",
        tier=Tier.TIER_3,
        is_opposite_field=True,
    ),
    Tournament(
        name="PGA Championship",
        date=date(2026, 5, 14),
        purse=18_000_000,
        course="Aronimink Golf Club",
        tier=Tier.TIER_1,
        is_major=True,
    ),
    Tournament(
        name="CJ Cup Honoring Byron Nelson",
        date=date(2026, 5, 21),
        purse=10_300_000,
        course="TPC Craig Ranch",
        tier=Tier.TIER_2,
    ),
    Tournament(
        name="Charles Schwab Challenge",
        date=date(2026, 5, 28),
        purse=9_900_000,
        course="Colonial Country Club",
        tier=Tier.TIER_2,
    ),

    # June
    Tournament(
        name="The Memorial Tournament",
        date=date(2026, 6, 4),
        purse=20_000_000,
        course="Muirfield Village",
        tier=Tier.TIER_1,
        is_signature=True,
        cut_rule=CutRule.TOP_50_TIES,  # 36-hole cut to top 50+ties
        field_size=SIGNATURE_FIELD_SIZE,
    ),
    Tournament(
        name="RBC Canadian Open",
        date=date(2026, 6, 11),
        purse=9_800_000,
        course="TPC Toronto",
        tier=Tier.TIER_2,
    ),
    Tournament(
        name="U.S. Open",
        date=date(2026, 6, 18),
        purse=21_000_000,
        course="Shinnecock Hills",
        tier=Tier.TIER_1,
        is_major=True,
    ),
    Tournament(
        name="Travelers Championship",
        date=date(2026, 6, 25),
        purse=20_000_000,
        course="TPC River Highlands",
        tier=Tier.TIER_1,
        is_signature=True,
        cut_rule=CutRule.NO_CUT,  # No-cut signature event
        field_size=SIGNATURE_FIELD_SIZE,
    ),

    # July
    Tournament(
        name="John Deere Classic",
        date=date(2026, 7, 2),
        purse=8_800_000,
        course="TPC Deere Run",
        tier=Tier.TIER_3,
    ),
    Tournament(
        name="Genesis Scottish Open",
        date=date(2026, 7, 9),
        purse=9_000_000,
        course="Renaissance Club",
        tier=Tier.TIER_2,
    ),
    Tournament(
        name="ISCO Championship",
        date=date(2026, 7, 9),
        purse=4_000_000,
        course="Hurstbourne Country Club",
        tier=Tier.TIER_3,
        is_opposite_field=True,
    ),
    Tournament(
        name="The Open Championship",
        date=date(2026, 7, 16),
        purse=17_000_000,
        course="Royal Birkdale",
        tier=Tier.TIER_1,
        is_major=True,
    ),
    Tournament(
        name="Corales Puntacana Championship",
        date=date(2026, 7, 16),
        purse=4_000_000,
        course="Puntacana Resort",
        tier=Tier.TIER_3,
        is_opposite_field=True,
    ),
    Tournament(
        name="3M Open",
        date=date(2026, 7, 23),
        purse=8_800_000,
        course="TPC Twin Cities",
        tier=Tier.TIER_3,
    ),
    Tournament(
        name="Rocket Mortgage Classic",
        date=date(2026, 7, 30),
        purse=10_000_000,
        course="Detroit Golf Club",
        tier=Tier.TIER_2,
    ),

    # August
    Tournament(
        name="Wyndham Championship",
        date=date(2026, 8, 6),
        purse=8_500_000,
        course="Sedgefield Country Club",
        tier=Tier.TIER_3,
    ),
    Tournament(
        name="FedEx St. Jude Championship",
        date=date(2026, 8, 13),
        purse=20_000_000,
        course="TPC Southwind",
        tier=Tier.TIER_1,
        is_playoff=True,
    ),
    Tournament(
        name="BMW Championship",
        date=date(2026, 8, 20),
        purse=20_000_000,
        course="Bellerive Country Club",
        tier=Tier.TIER_1,
        is_playoff=True,
    ),
    Tournament(
        name="Tour Championship",
        date=date(2026, 8, 27),
        purse=40_000_000,
        course="East Lake Golf Club",
        tier=Tier.TIER_1,
        is_playoff=True,
        cut_rule=CutRule.NO_CUT,  # Top 30 only, no cut
        field_size=30,
    ),

    # FedEx Cup Fall Events (2026 purses)
    Tournament(
        name="Procore Championship",
        date=date(2026, 9, 10),
        purse=6_000_000,
        course="Silverado Resort",
        tier=Tier.TIER_3,
    ),
    Tournament(
        name="Sanderson Farms Championship",
        date=date(2026, 9, 24),
        purse=6_000_000,
        course="Country Club of Jackson",
        tier=Tier.TIER_3,
    ),
    Tournament(
        name="Shriners Children's Open",
        date=date(2026, 10, 8),
        purse=6_000_000,
        course="TPC Summerlin",
        tier=Tier.TIER_3,
    ),
    Tournament(
        name="ZOZO Championship",
        date=date(2026, 10, 15),
        purse=6_000_000,
        course="Accordia Golf Narashino",
        tier=Tier.TIER_3,
    ),
    Tournament(
        name="World Wide Technology Championship",
        date=date(2026, 10, 29),
        purse=6_000_000,
        course="El Cardonal at Diamante",
        tier=Tier.TIER_3,
    ),
    Tournament(
        name="Butterfield Bermuda Championship",
        date=date(2026, 11, 5),
        purse=6_000_000,
        course="Port Royal Golf Course",
        tier=Tier.TIER_3,
    ),
    Tournament(
        name="Baycurrent Classic",
        date=date(2026, 11, 12),
        purse=8_000_000,
        course="TBD",
        tier=Tier.TIER_2,
    ),
    Tournament(
        name="The RSM Classic",
        date=date(2026, 11, 19),
        purse=7_400_000,
        course="Sea Island Golf Club",
        tier=Tier.TIER_3,
    ),
    Tournament(
        name="Biltmore Championship Asheville",
        date=date(2026, 12, 3),
        purse=5_000_000,
        course="Biltmore Forest Country Club",
        tier=Tier.TIER_3,
    ),
]


def get_schedule() -> List[Tournament]:
    """Get the 2026 PGA Tour schedule."""
    return SCHEDULE_2026.copy()


def get_tournament_by_name(name: str) -> Optional[Tournament]:
    """Find tournament by name (case-insensitive partial match)."""
    name_lower = name.lower()
    for t in SCHEDULE_2026:
        if name_lower in t.name.lower():
            return t
    return None


def get_tournament_by_date(target_date: date) -> List[Tournament]:
    """Get tournaments on a specific date."""
    return [t for t in SCHEDULE_2026 if t.date == target_date]


def get_upcoming_tournaments(from_date: Optional[date] = None) -> List[Tournament]:
    """Get tournaments from a date forward, sorted by date."""
    from_date = from_date or date.today()
    upcoming = [t for t in SCHEDULE_2026 if t.date >= from_date]
    return sorted(upcoming, key=lambda t: t.date)


def get_next_tournament(from_date: Optional[date] = None) -> Optional[Tournament]:
    """Get the next tournament from a date."""
    upcoming = get_upcoming_tournaments(from_date)
    return upcoming[0] if upcoming else None


def get_majors() -> List[Tournament]:
    """Get all major tournaments."""
    return [t for t in SCHEDULE_2026 if t.is_major]


def get_signature_events() -> List[Tournament]:
    """Get all signature events (Tier 1 non-majors)."""
    return [t for t in SCHEDULE_2026 if t.is_signature]


def get_tier_1_events() -> List[Tournament]:
    """Get all Tier 1 events ($20M+ purse)."""
    return [t for t in SCHEDULE_2026 if t.tier == Tier.TIER_1]


def get_tier_3_events() -> List[Tournament]:
    """Get all Tier 3 events (lower purse, easier fields)."""
    return [t for t in SCHEDULE_2026 if t.tier == Tier.TIER_3]


def get_playoff_events() -> List[Tournament]:
    """Get FedEx playoff events."""
    return [t for t in SCHEDULE_2026 if t.is_playoff]


def get_no_cut_events() -> List[Tournament]:
    """Get all no-cut events (guaranteed earnings)."""
    return [t for t in SCHEDULE_2026 if t.cut_rule == CutRule.NO_CUT]


def get_fall_events() -> List[Tournament]:
    """Get FedEx Cup Fall events (September-December)."""
    return [t for t in SCHEDULE_2026 if t.date.month >= 9]


def get_total_purse() -> int:
    """Get total prize money for the season."""
    return sum(t.purse for t in SCHEDULE_2026)


# CSS selectors for scraping (can be updated if site changes)
SCRAPER_SELECTORS: Dict[str, str] = {
    "login_email": "input[name='email'], input[type='email']",
    "login_password": "input[name='password'], input[type='password']",
    "login_button": "button[type='submit'], input[type='submit']",
    "standings_table": "table.standings, .standings-table, #standings",
    "standings_row": "tr, .standing-row",
    "golfer_select": "select.golfer-select, #golfer-select",
    "golfer_option": "option",
    "picks_table": "table.picks, .picks-table, #picks",
    "schedule_table": "table.schedule, .schedule-table, #schedule",
}


# Payout distribution (position -> percentage of purse)
PAYOUT_DISTRIBUTION: Dict[int, float] = {
    1: 0.180, 2: 0.109, 3: 0.069, 4: 0.049, 5: 0.041,
    6: 0.036, 7: 0.0335, 8: 0.031, 9: 0.029, 10: 0.027,
    11: 0.025, 12: 0.023, 13: 0.0215, 14: 0.020, 15: 0.0195,
    16: 0.019, 17: 0.0185, 18: 0.018, 19: 0.0175, 20: 0.017,
    21: 0.0165, 22: 0.016, 23: 0.0155, 24: 0.015, 25: 0.0145,
    26: 0.014, 27: 0.0135, 28: 0.013, 29: 0.0125, 30: 0.012,
    31: 0.0115, 32: 0.011, 33: 0.0105, 34: 0.010, 35: 0.00975,
    36: 0.0095, 37: 0.00925, 38: 0.009, 39: 0.00875, 40: 0.0085,
    41: 0.00825, 42: 0.008, 43: 0.00775, 44: 0.0075, 45: 0.00725,
    46: 0.007, 47: 0.00675, 48: 0.0065, 49: 0.00625, 50: 0.006,
    51: 0.00585, 52: 0.0057, 53: 0.00555, 54: 0.0054, 55: 0.00535,
    56: 0.0053, 57: 0.00525, 58: 0.0052, 59: 0.00515, 60: 0.0051,
    61: 0.00505, 62: 0.005, 63: 0.00495, 64: 0.0049, 65: 0.00485,
}
