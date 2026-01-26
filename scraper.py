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

try:
    from .config import get_config, SCRAPER_SELECTORS
    from .database import Database
    from .models import LeagueStanding, OpponentPick, Pick
except ImportError:
    from config import get_config, SCRAPER_SELECTORS
    from database import Database
    from models import LeagueStanding, OpponentPick, Pick

logger = logging.getLogger(__name__)


class Scraper:
    """Web scraper for buzzfantasygolf.com."""

    # League-specific URLs (Bushwood league)
    LEAGUE_ID = "25378"
    BASE_URL = "https://www.buzzfantasygolf.com"

    def __init__(self, headless: bool = True):
        """Initialize scraper."""
        self.config = get_config()
        self.db = Database()
        self.headless = headless
        self._driver: Optional[webdriver.Chrome] = None
        self._request_count = 0
        self._logged_in = False
        self._team_id: Optional[str] = None

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
                "#Email",
                "input[name='Email']",
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
                "#Password",
                "input[name='Password']",
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
            # Navigate to league standings page
            url = f"{self.BASE_URL}/leagues/{self.LEAGUE_ID}/standings"
            driver.get(url)
            self._wait_and_throttle()

            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )
                html = driver.page_source
            except TimeoutException:
                logger.error("Could not find standings table")
                return []

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

        # Find the standings table
        table = soup.find("table")
        if not table:
            logger.warning("No standings table found in HTML")
            return []

        rows = table.find_all("tr")[1:]  # Skip header row
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 3:
                continue

            try:
                # Table format: Place | Team Name (with username) | Earnings
                place_text = cells[0].get_text(strip=True)
                # Handle "T2" format for ties
                rank = int(place_text.replace("T", "").strip()) if place_text else 0

                # Team name cell contains both team name and username
                name_cell = cells[1].get_text(separator="\n", strip=True)
                lines = [l.strip() for l in name_cell.split("\n") if l.strip()]
                username = lines[0] if lines else ""
                player_name = lines[1] if len(lines) > 1 else username

                # Earnings
                earnings_text = cells[2].get_text(strip=True)
                earnings_str = earnings_text.replace("$", "").replace(",", "")
                earnings = int(float(earnings_str)) if earnings_str else 0

                standings.append(LeagueStanding(
                    rank=rank,
                    player_name=player_name,
                    username=username,
                    total_earnings=earnings,
                    cuts_made=0,
                    picks_made=0,
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
            # Navigate to current field page which shows available golfers
            url = f"{self.BASE_URL}/leagues/{self.LEAGUE_ID}/field"
            driver.get(url)
            self._wait_and_throttle()
            time.sleep(3)

            # Wait for table to load
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )
            except TimeoutException:
                pass

            html = driver.page_source
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
        """Parse available golfers from field/selection page."""
        soup = BeautifulSoup(html, "html.parser")
        golfers = []

        # Try table format first (field page has a table of golfers)
        table = soup.find("table")
        if table:
            rows = table.find_all("tr")[1:]  # Skip header
            for row in rows:
                cells = row.find_all("td")
                if cells:
                    # First cell usually contains golfer name
                    name = cells[0].get_text(strip=True)
                    if name and name not in ["", "Golfer Name", "Select"]:
                        golfers.append(name)

        # Also try select/dropdown
        if not golfers:
            select = soup.find("select")
            if select:
                options = select.find_all("option")
                for opt in options:
                    name = opt.get_text(strip=True)
                    if name and name not in ["Select a golfer", "--", ""]:
                        golfers.append(name)

        # Also try list/div format
        if not golfers:
            golfer_divs = soup.find_all(class_=lambda c: c and "golfer" in c.lower() if c else False)
            for div in golfer_divs:
                name = div.get_text(strip=True)
                if name:
                    golfers.append(name)

        logger.info(f"Found {len(golfers)} available golfers")
        return golfers

    def get_opponent_picks(self, tournament_name: str = "") -> List[OpponentPick]:
        """Scrape all opponent picks from live scoring page."""
        cache_key = f"scraper:opponent_picks:{tournament_name}"
        cached = self.db.get_cache(cache_key)
        if cached:
            return [OpponentPick(**p) for p in cached]

        if not self._logged_in and not self.login():
            return []

        driver = self._get_driver()
        try:
            # Use live scoring page which shows all teams and their picks
            url = f"{self.BASE_URL}/leagues/{self.LEAGUE_ID}/livescoring"
            driver.get(url)
            self._wait_and_throttle()

            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "table"))
                )
            except TimeoutException:
                pass

            time.sleep(2)
            html = driver.page_source
            picks = self._parse_opponent_picks(html, tournament_name)

            # Cache and save
            if picks:
                expires = datetime.now() + timedelta(hours=self.config.cache_expiry_hours)
                self.db.set_cache(cache_key, [
                    {
                        "opponent_username": p.opponent_username,
                        "golfer_name": p.golfer_name,
                        "tournament_name": p.tournament_name,
                        "tournament_date": p.tournament_date.isoformat() if hasattr(p.tournament_date, 'isoformat') else str(p.tournament_date),
                    } for p in picks
                ], expires)

                for pick in picks:
                    self.db.save_opponent_pick(pick)

            return picks

        except Exception as e:
            logger.error(f"Failed to get opponent picks: {e}")
            return []

    def _parse_opponent_picks(self, html: str, filter_tournament: str = "") -> List[OpponentPick]:
        """Parse opponent picks from live scoring HTML."""
        soup = BeautifulSoup(html, "html.parser")
        picks = []

        # Live scoring table format: Pos | Chg | Team Name | Golfer | Earnings
        table = soup.find("table")
        if not table:
            return picks

        rows = table.find_all("tr")[1:]  # Skip header
        for row in rows:
            cells = row.find_all("td")
            if len(cells) < 4:
                continue

            try:
                # Format: Pos | Chg | Team Name (with username) | Golfer | Earnings
                team_cell = cells[2].get_text(separator="\n", strip=True)
                lines = [l.strip() for l in team_cell.split("\n") if l.strip()]
                username = lines[0] if lines else ""

                golfer = cells[3].get_text(strip=True)

                if username and golfer:
                    picks.append(OpponentPick(
                        opponent_username=username,
                        golfer_name=golfer,
                        tournament_name=filter_tournament or "Current Tournament",
                        tournament_date=date.today(),
                    ))
            except Exception as e:
                logger.warning(f"Failed to parse pick row: {e}")
                continue

        logger.info(f"Parsed {len(picks)} opponent picks")
        return picks

    def _get_my_team_id(self) -> Optional[str]:
        """Get the user's team ID from the league page."""
        if self._team_id:
            return self._team_id

        driver = self._get_driver()
        try:
            driver.get(f"{self.BASE_URL}/leagues/{self.LEAGUE_ID}/home")
            self._wait_and_throttle()

            # Find the "View Team" link which contains the team ID
            links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/teams/']")
            for link in links:
                href = link.get_attribute("href") or ""
                if "/teams/" in href and self.LEAGUE_ID in href:
                    # Extract team ID from URL like .../teams/176808
                    parts = href.split("/teams/")
                    if len(parts) > 1:
                        self._team_id = parts[1].split("/")[0].split("?")[0]
                        logger.info(f"Found team ID: {self._team_id}")
                        return self._team_id
        except Exception as e:
            logger.warning(f"Could not find team ID: {e}")

        return None

    def get_my_picks(self) -> List[Pick]:
        """Scrape user's own picks history from team page."""
        if not self._logged_in and not self.login():
            return []

        team_id = self._get_my_team_id()
        if not team_id:
            logger.error("Could not find team ID")
            return []

        driver = self._get_driver()
        picks = []

        try:
            # Get list of tournaments from league page
            driver.get(f"{self.BASE_URL}/leagues/{self.LEAGUE_ID}/tournaments")
            self._wait_and_throttle()
            time.sleep(2)

            # Find tournament links
            tournament_links = []
            links = driver.find_elements(By.CSS_SELECTOR, f"a[href*='/leagues/{self.LEAGUE_ID}/tournaments/']")
            for link in links:
                href = link.get_attribute("href") or ""
                text = link.text.strip()
                if text and "/tournaments/" in href:
                    # Extract tournament ID
                    parts = href.split("/tournaments/")
                    if len(parts) > 1:
                        t_id = parts[1].split("/")[0].split("?")[0]
                        if t_id.isdigit():
                            tournament_links.append((t_id, text))

            logger.info(f"Found {len(tournament_links)} tournaments")

            # Go to team page and check each tournament
            driver.get(f"{self.BASE_URL}/leagues/{self.LEAGUE_ID}/teams/{team_id}")
            time.sleep(3)

            # Wait for dynamic content to load
            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "table"))
                )
            except TimeoutException:
                pass

            time.sleep(2)

            # Find the picks table (second table usually has picks)
            tables = driver.find_elements(By.TAG_NAME, "table")
            for table in tables:
                rows = table.find_elements(By.TAG_NAME, "tr")
                for row in rows[1:]:  # Skip header
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 2:
                        try:
                            # Format: Golfer Name | Place | R1 | R2 | R3 | R4 | Total | To Par | Earnings
                            golfer_cell = cells[0].text.strip()
                            # Extract golfer name (before "Ownership:")
                            golfer_name = golfer_cell.split("\n")[0].strip()
                            if not golfer_name or golfer_name == "Total:":
                                continue

                            position = None
                            earnings = 0

                            if len(cells) > 1:
                                pos_text = cells[1].text.strip()
                                if pos_text.isdigit():
                                    position = int(pos_text)

                            if len(cells) > 8:
                                earn_text = cells[8].text.strip().replace("$", "").replace(",", "")
                                try:
                                    earnings = int(float(earn_text))
                                except:
                                    pass

                            # We need to determine which tournament this is for
                            # For now, use a placeholder - will be updated from tournament selector
                            pick = Pick(
                                golfer_name=golfer_name,
                                tournament_name="Current Tournament",
                                tournament_date=date.today(),
                                earnings=earnings,
                                position=position,
                                made_cut=position is not None and position <= 65,
                            )
                            picks.append(pick)
                            logger.info(f"Found pick: {golfer_name} (pos: {position}, earnings: ${earnings})")

                        except Exception as e:
                            logger.warning(f"Failed to parse pick row: {e}")
                            continue

            # Also get picks from standings/history if available
            # by cross-referencing with livescoring page
            driver.get(f"{self.BASE_URL}/leagues/{self.LEAGUE_ID}/livescoring")
            time.sleep(2)

            table = driver.find_element(By.TAG_NAME, "table") if driver.find_elements(By.TAG_NAME, "table") else None
            if table:
                rows = table.find_elements(By.TAG_NAME, "tr")[1:]
                for row in rows:
                    cells = row.find_elements(By.TAG_NAME, "td")
                    if len(cells) >= 4:
                        team_name = cells[2].text.strip().lower()
                        if "gitberge" in team_name or "chip and a putt" in team_name:
                            golfer = cells[3].text.strip()
                            earnings_text = cells[4].text.strip() if len(cells) > 4 else "$0"
                            earnings = int(float(earnings_text.replace("$", "").replace(",", ""))) if "$" in earnings_text else 0

                            # Check if we already have this pick
                            existing = [p for p in picks if p.golfer_name == golfer]
                            if not existing:
                                pick = Pick(
                                    golfer_name=golfer,
                                    tournament_name="Current Tournament",
                                    tournament_date=date.today(),
                                    earnings=earnings,
                                    position=None,
                                    made_cut=True,
                                )
                                picks.append(pick)

            # Save picks to database
            for pick in picks:
                self.db.save_pick(pick)

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
