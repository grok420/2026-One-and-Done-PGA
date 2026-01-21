"""
Web scraper for buzzfantasygolf.com.
Extracts league standings, picks, available golfers, and schedule.
"""

import logging
import time
import re
from datetime import date, datetime, timedelta
from typing import List, Optional, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

from .config import get_config, SCRAPER_SELECTORS
from .database import Database
from .models import LeagueStanding, OpponentPick, Pick

logger = logging.getLogger(__name__)


class Scraper:
    """Web scraper for buzzfantasygolf.com."""

    def __init__(self, headless: bool = True):
        """Initialize scraper."""
        self.config = get_config()
        self.db = Database()
        self.headless = headless
        self._driver: Optional[webdriver.Chrome] = None
        self._request_count = 0
        self._logged_in = False

    def _get_driver(self) -> webdriver.Chrome:
        """Get or create Chrome WebDriver."""
        if self._driver is None:
            options = Options()
            if self.headless:
                options.add_argument("--headless=new")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--disable-gpu")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36")

            service = Service(ChromeDriverManager().install())
            self._driver = webdriver.Chrome(service=service, options=options)
            self._driver.implicitly_wait(10)
        return self._driver

    def _wait_and_throttle(self):
        """Enforce rate limiting."""
        self._request_count += 1
        if self._request_count > self.config.max_requests_per_session:
            logger.warning("Rate limit reached, waiting...")
            time.sleep(60)
            self._request_count = 0
        else:
            time.sleep(self.config.request_delay_seconds)

    def _retry(self, func, max_retries: int = None, *args, **kwargs):
        """Retry a function with exponential backoff."""
        max_retries = max_retries or self.config.max_retries
        last_error = None
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                wait_time = (2 ** attempt) * self.config.request_delay_seconds
                logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
        raise last_error

    def login(self) -> bool:
        """Login to buzzfantasygolf.com."""
        if self._logged_in:
            return True

        driver = self._get_driver()
        try:
            logger.info("Logging in to buzzfantasygolf.com...")
            driver.get(f"{self.config.site_base_url}/login")
            self._wait_and_throttle()

            # Wait for login form
            wait = WebDriverWait(driver, 20)

            # Try multiple selectors for email field
            email_selectors = [
                "input[name='email']",
                "input[type='email']",
                "#email",
                "input[placeholder*='email' i]",
            ]
            email_input = None
            for selector in email_selectors:
                try:
                    email_input = wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                    break
                except TimeoutException:
                    continue

            if not email_input:
                logger.error("Could not find email input field")
                return False

            # Find password field
            password_selectors = [
                "input[name='password']",
                "input[type='password']",
                "#password",
            ]
            password_input = None
            for selector in password_selectors:
                try:
                    password_input = driver.find_element(By.CSS_SELECTOR, selector)
                    break
                except NoSuchElementException:
                    continue

            if not password_input:
                logger.error("Could not find password input field")
                return False

            # Enter credentials
            email_input.clear()
            email_input.send_keys(self.config.site_email)
            password_input.clear()
            password_input.send_keys(self.config.site_password)

            # Find and click submit button
            submit_selectors = [
                "button[type='submit']",
                "input[type='submit']",
                "button:contains('Login')",
                ".login-button",
                "#login-btn",
            ]
            for selector in submit_selectors:
                try:
                    submit_btn = driver.find_element(By.CSS_SELECTOR, selector)
                    submit_btn.click()
                    break
                except NoSuchElementException:
                    continue

            # Wait for login to complete (URL change or dashboard element)
            time.sleep(3)
            self._wait_and_throttle()

            # Check if logged in
            if "login" not in driver.current_url.lower():
                self._logged_in = True
                logger.info("Login successful")
                return True

            logger.error("Login may have failed - still on login page")
            return False

        except Exception as e:
            logger.error(f"Login failed: {e}")
            return False

    def get_standings(self) -> List[LeagueStanding]:
        """Scrape league standings."""
        # Check cache first
        cache_key = "scraper:standings"
        cached = self.db.get_cache(cache_key)
        if cached:
            logger.info("Using cached standings")
            return [LeagueStanding(**s) for s in cached]

        if not self._logged_in and not self.login():
            logger.error("Must be logged in to get standings")
            return []

        driver = self._get_driver()
        try:
            # Navigate to league/standings page
            standings_urls = [
                f"{self.config.site_base_url}/league/{self.config.league_name}/standings",
                f"{self.config.site_base_url}/standings",
                f"{self.config.site_base_url}/league/standings",
            ]

            html = None
            for url in standings_urls:
                try:
                    driver.get(url)
                    self._wait_and_throttle()
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "table"))
                    )
                    html = driver.page_source
                    break
                except TimeoutException:
                    continue

            if not html:
                logger.error("Could not find standings page")
                return []

            standings = self._parse_standings(html)

            # Cache results
            if standings:
                expires = datetime.now() + timedelta(hours=self.config.cache_expiry_hours)
                self.db.set_cache(cache_key, [
                    {
                        "rank": s.rank,
                        "player_name": s.player_name,
                        "username": s.username,
                        "total_earnings": s.total_earnings,
                        "cuts_made": s.cuts_made,
                        "picks_made": s.picks_made,
                        "majors_earnings": s.majors_earnings,
                    } for s in standings
                ], expires)

                # Also save to database
                self.db.save_standings(standings)

            return standings

        except Exception as e:
            logger.error(f"Failed to get standings: {e}")
            return []

    def _parse_standings(self, html: str) -> List[LeagueStanding]:
        """Parse standings from HTML."""
        soup = BeautifulSoup(html, "html.parser")
        standings = []

        # Try different table selectors
        table = None
        for selector in ["table.standings", ".standings-table", "#standings", "table"]:
            table = soup.select_one(selector)
            if table:
                break

        if not table:
            logger.warning("No standings table found in HTML")
            return []

        rows = table.find_all("tr")[1:]  # Skip header row
        for i, row in enumerate(rows):
            cells = row.find_all(["td", "th"])
            if len(cells) < 3:
                continue

            try:
                # Extract data - format varies by site
                rank = i + 1
                player_name = ""
                username = ""
                earnings = 0
                cuts_made = 0

                for j, cell in enumerate(cells):
                    text = cell.get_text(strip=True)
                    # First column usually rank or name
                    if j == 0:
                        if text.isdigit():
                            rank = int(text)
                        else:
                            player_name = text
                    elif j == 1:
                        if not player_name:
                            player_name = text
                        else:
                            username = text
                    elif "$" in text or text.replace(",", "").replace(".", "").isdigit():
                        # This is likely earnings
                        earnings_str = text.replace("$", "").replace(",", "")
                        try:
                            earnings = int(float(earnings_str))
                        except ValueError:
                            pass
                    elif text.isdigit() and int(text) < 50:
                        # Likely cuts made or picks count
                        cuts_made = int(text)

                # Use player_name as username if not found
                if not username:
                    username = player_name.lower().replace(" ", "")

                standings.append(LeagueStanding(
                    rank=rank,
                    player_name=player_name,
                    username=username,
                    total_earnings=earnings,
                    cuts_made=cuts_made,
                    picks_made=0,  # May not be on standings page
                    majors_earnings=0,
                ))
            except Exception as e:
                logger.warning(f"Failed to parse standings row: {e}")
                continue

        logger.info(f"Parsed {len(standings)} standings entries")
        return standings

    def get_available_golfers(self) -> List[str]:
        """Scrape list of available (unused) golfers for user."""
        cache_key = "scraper:available_golfers"
        cached = self.db.get_cache(cache_key)
        if cached:
            logger.info("Using cached available golfers")
            return cached

        if not self._logged_in and not self.login():
            return []

        driver = self._get_driver()
        try:
            # Navigate to pick/selection page
            pick_urls = [
                f"{self.config.site_base_url}/pick",
                f"{self.config.site_base_url}/make-pick",
                f"{self.config.site_base_url}/selection",
            ]

            html = None
            for url in pick_urls:
                try:
                    driver.get(url)
                    self._wait_and_throttle()
                    time.sleep(2)  # Wait for JS to load
                    html = driver.page_source
                    break
                except Exception:
                    continue

            if not html:
                return []

            golfers = self._parse_available_golfers(html)

            # Cache and save to database
            if golfers:
                expires = datetime.now() + timedelta(hours=self.config.cache_expiry_hours)
                self.db.set_cache(cache_key, golfers, expires)
                self.db.save_available_golfers(golfers)

            return golfers

        except Exception as e:
            logger.error(f"Failed to get available golfers: {e}")
            return []

    def _parse_available_golfers(self, html: str) -> List[str]:
        """Parse available golfers from selection page."""
        soup = BeautifulSoup(html, "html.parser")
        golfers = []

        # Try select/dropdown first
        select = soup.find("select", class_=lambda c: c and "golfer" in c.lower() if c else False)
        if not select:
            select = soup.find("select", id=lambda i: i and "golfer" in i.lower() if i else False)
        if not select:
            select = soup.find("select")

        if select:
            options = select.find_all("option")
            for opt in options:
                name = opt.get_text(strip=True)
                if name and name != "Select a golfer" and name != "--":
                    golfers.append(name)
        else:
            # Try list/div format
            golfer_divs = soup.find_all(class_=lambda c: c and "golfer" in c.lower() if c else False)
            for div in golfer_divs:
                name = div.get_text(strip=True)
                if name:
                    golfers.append(name)

        logger.info(f"Found {len(golfers)} available golfers")
        return golfers

    def get_opponent_picks(self, tournament_name: str = "") -> List[OpponentPick]:
        """Scrape all opponent picks for a tournament."""
        cache_key = f"scraper:opponent_picks:{tournament_name}"
        cached = self.db.get_cache(cache_key)
        if cached:
            return [OpponentPick(**p) for p in cached]

        if not self._logged_in and not self.login():
            return []

        driver = self._get_driver()
        try:
            # Navigate to picks/history page
            picks_urls = [
                f"{self.config.site_base_url}/league/{self.config.league_name}/picks",
                f"{self.config.site_base_url}/picks",
                f"{self.config.site_base_url}/league/picks",
            ]

            html = None
            for url in picks_urls:
                try:
                    driver.get(url)
                    self._wait_and_throttle()
                    WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.TAG_NAME, "table"))
                    )
                    html = driver.page_source
                    break
                except TimeoutException:
                    continue

            if not html:
                return []

            picks = self._parse_opponent_picks(html, tournament_name)

            # Cache and save
            if picks:
                expires = datetime.now() + timedelta(hours=self.config.cache_expiry_hours)
                self.db.set_cache(cache_key, [
                    {
                        "opponent_username": p.opponent_username,
                        "golfer_name": p.golfer_name,
                        "tournament_name": p.tournament_name,
                        "tournament_date": p.tournament_date.isoformat(),
                    } for p in picks
                ], expires)

                for pick in picks:
                    self.db.save_opponent_pick(pick)

            return picks

        except Exception as e:
            logger.error(f"Failed to get opponent picks: {e}")
            return []

    def _parse_opponent_picks(self, html: str, filter_tournament: str = "") -> List[OpponentPick]:
        """Parse opponent picks from HTML."""
        soup = BeautifulSoup(html, "html.parser")
        picks = []

        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")[1:]  # Skip header
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) < 2:
                    continue

                try:
                    # Try to extract username, golfer, tournament
                    username = ""
                    golfer = ""
                    tournament = ""
                    tournament_date = date.today()

                    for cell in cells:
                        text = cell.get_text(strip=True)
                        # Identify field type by content/position
                        if "@" in text or len(text) < 20:
                            if not username:
                                username = text
                        elif any(name in text.lower() for name in ["open", "championship", "classic", "invitational"]):
                            tournament = text
                        else:
                            # Assume it's a golfer name
                            if not golfer:
                                golfer = text

                    if username and golfer:
                        if filter_tournament and tournament and filter_tournament.lower() not in tournament.lower():
                            continue

                        picks.append(OpponentPick(
                            opponent_username=username,
                            golfer_name=golfer,
                            tournament_name=tournament or "Unknown",
                            tournament_date=tournament_date,
                        ))
                except Exception as e:
                    logger.warning(f"Failed to parse pick row: {e}")
                    continue

        logger.info(f"Parsed {len(picks)} opponent picks")
        return picks

    def get_my_picks(self) -> List[Pick]:
        """Scrape user's own picks history."""
        if not self._logged_in and not self.login():
            return []

        driver = self._get_driver()
        try:
            driver.get(f"{self.config.site_base_url}/my-picks")
            self._wait_and_throttle()
            time.sleep(2)

            html = driver.page_source
            soup = BeautifulSoup(html, "html.parser")
            picks = []

            table = soup.find("table")
            if not table:
                return []

            rows = table.find_all("tr")[1:]
            for row in rows:
                cells = row.find_all(["td", "th"])
                if len(cells) < 2:
                    continue

                try:
                    golfer = ""
                    tournament = ""
                    earnings = 0
                    position = None
                    made_cut = False

                    for cell in cells:
                        text = cell.get_text(strip=True)
                        if "$" in text:
                            earnings_str = text.replace("$", "").replace(",", "")
                            try:
                                earnings = int(float(earnings_str))
                            except ValueError:
                                pass
                        elif text.isdigit():
                            pos = int(text)
                            if pos < 100:
                                position = pos
                                made_cut = pos <= 70
                        elif any(x in text.lower() for x in ["open", "championship", "classic"]):
                            tournament = text
                        else:
                            if not golfer:
                                golfer = text

                    if golfer and tournament:
                        pick = Pick(
                            golfer_name=golfer,
                            tournament_name=tournament,
                            tournament_date=date.today(),  # Would need to parse
                            earnings=earnings,
                            position=position,
                            made_cut=made_cut,
                        )
                        picks.append(pick)
                        self.db.save_pick(pick)
                except Exception as e:
                    logger.warning(f"Failed to parse my pick: {e}")
                    continue

            return picks

        except Exception as e:
            logger.error(f"Failed to get my picks: {e}")
            return []

    def refresh_all_data(self) -> Dict[str, int]:
        """Refresh all data from site."""
        results = {}

        if not self.login():
            return {"error": "Login failed"}

        # Fetch standings
        standings = self.get_standings()
        results["standings"] = len(standings)

        # Fetch available golfers
        available = self.get_available_golfers()
        results["available_golfers"] = len(available)

        # Fetch opponent picks
        picks = self.get_opponent_picks()
        results["opponent_picks"] = len(picks)

        # Fetch my picks
        my_picks = self.get_my_picks()
        results["my_picks"] = len(my_picks)

        return results

    def close(self):
        """Close the browser."""
        if self._driver:
            self._driver.quit()
            self._driver = None
            self._logged_in = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_scraper(headless: bool = True) -> Scraper:
    """Get configured scraper instance."""
    return Scraper(headless=headless)
