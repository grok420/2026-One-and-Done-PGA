#!/usr/bin/env python3
"""
Automatic sync script for PGA One and Done.
Imports picks, standings, and opponent data from buzzfantasygolf.com.

Run manually: python auto_sync.py
Schedule with cron: 0 8 * * * cd ~/pga_one_and_done && ./venv/bin/python auto_sync.py
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from scraper import Scraper
from database import Database

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent / 'sync.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def run_sync():
    """Run full data sync from fantasy site."""
    logger.info("=" * 50)
    logger.info(f"Starting auto-sync at {datetime.now()}")

    db = Database()

    try:
        with Scraper(headless=True) as scraper:
            # Login
            if not scraper.login():
                logger.error("Login failed!")
                return False

            logger.info("Login successful")

            # Sync standings
            logger.info("Fetching standings...")
            standings = scraper.get_standings()
            if standings:
                db.save_standings(standings)
                logger.info(f"Saved {len(standings)} standings")

            # Sync my picks
            logger.info("Fetching my picks...")
            my_picks = scraper.get_my_picks()
            if my_picks:
                for pick in my_picks:
                    db.save_pick(pick)
                logger.info(f"Saved {len(my_picks)} picks")

            # Sync available golfers
            logger.info("Fetching available golfers...")
            available = scraper.get_available_golfers()
            if available:
                db.save_available_golfers(available)
                logger.info(f"Saved {len(available)} available golfers")

            # Sync opponent picks
            logger.info("Fetching opponent picks...")
            opponent_picks = scraper.get_opponent_picks()
            if opponent_picks:
                for pick in opponent_picks:
                    db.save_opponent_pick(pick)
                logger.info(f"Saved {len(opponent_picks)} opponent picks")

            logger.info("Sync completed successfully!")
            return True

    except Exception as e:
        logger.error(f"Sync failed: {e}")
        return False


if __name__ == "__main__":
    success = run_sync()
    sys.exit(0 if success else 1)
