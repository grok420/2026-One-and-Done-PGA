# PGA One and Done Optimizer

A comprehensive fantasy golf optimization tool for One and Done leagues on BuzzFantasyGolf.com. Uses Monte Carlo simulations, Data Golf API predictions, and game theory-based strategy to maximize your expected value.

## Features

- **Monte Carlo Simulations**: Run 50,000+ simulations per golfer to estimate expected value
- **Data Golf API Integration**: Real-time predictions, win probabilities, and strokes gained data
- **Web Scraping**: Automatic extraction of league standings, opponent picks, and available golfers
- **Game Theory Strategy**: EV-hedged allocation with opponent usage tracking
- **Beautiful CLI**: Rich terminal interface with colorful tables and progress indicators
- **Streamlit Web App**: Interactive web UI accessible remotely
- **Full 2026 Schedule**: All 39 tournaments with official purses hardcoded

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/gitberge/2026-One-and-Done-PGA.git
cd 2026-One-and-Done-PGA

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

### Configuration

```bash
# Copy the environment template
cp .env.template .env

# Edit .env with your credentials
nano .env
```

Required environment variables:
- `PGA_OAD_EMAIL`: Your BuzzFantasyGolf email
- `PGA_OAD_PASSWORD`: Your BuzzFantasyGolf password
- `PGA_OAD_USERNAME`: Your BuzzFantasyGolf username
- `DATAGOLF_API_KEY`: Your Data Golf API key

### Usage

#### Command Line Interface

```bash
# First-time setup
pga-oad setup

# Get pick recommendations
pga-oad recommend

# View league standings
pga-oad standings

# View tournament schedule
pga-oad schedule

# Run what-if analysis
pga-oad whatif "Scottie Scheffler"

# Generate season plan
pga-oad plan

# Run Monte Carlo simulation
pga-oad simulate "Rory McIlroy" -t "The Masters"
```

#### Web Interface (Streamlit)

```bash
# Run locally
streamlit run pga_one_and_done/web_app.py

# Run for remote access
streamlit run pga_one_and_done/web_app.py --server.port 8501 --server.address 0.0.0.0
```

Then open http://localhost:8501 in your browser.

## Strategy Overview

The optimizer implements "Grok's EV-Hedged Allocation" strategy:

### Phase-Based Approach

1. **Early Season (Jan-Mar)**: Use mid-tier golfers for Tier 2/3 events. Save elites for later.
2. **Mid Season (Apr-Jul)**: Deploy elites for majors and signature events.
3. **Playoffs (Aug)**: Use remaining elites for FedEx Cup events.

### Key Concepts

- **Expected Value (EV)**: Primary metric for ranking picks
- **Hedge Bonus**: Bonus for picking golfers unused by opponents
- **Regret Minimization**: Consider opportunity cost vs alternatives
- **Tier Classification**:
  - Tier 1: $20M+ purse (majors, signatures)
  - Tier 2: $9-15M purse (regular events)
  - Tier 3: <$9M purse (opposite-field events)

## Project Structure

```
pga_one_and_done/
├── __init__.py          # Package initialization
├── cli.py               # Command-line interface (Click + Rich)
├── config.py            # Configuration and 2026 schedule
├── database.py          # SQLite persistence layer
├── api.py               # Data Golf API client
├── scraper.py           # Selenium web scraper
├── simulator.py         # Monte Carlo simulation engine
├── strategy.py          # Recommendation engine
├── models.py            # Data classes
├── web_app.py           # Streamlit web interface
├── requirements.txt     # Dependencies
└── setup.py             # Package setup
```

## 2026 Tournament Highlights

| Event | Purse | Tier |
|-------|-------|------|
| The Players Championship | $25,000,000 | Tier 1 |
| U.S. Open | $21,000,000 | Tier 1 (Major) |
| The Masters | $20,000,000 | Tier 1 (Major) |
| Tour Championship | $40,000,000 | Tier 1 (Playoff) |

Total season purse: ~$450,000,000+

## API Endpoints Used

Data Golf API:
- `/preds/pre-tournament`: Win/top-10/cut probabilities
- `/preds/skill-ratings`: Strokes gained statistics
- `/get-player-list`: Full player database
- `/field-updates`: Current tournament field

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Data Golf for their excellent prediction API
- BuzzFantasyGolf for hosting the league
- Streamlit for the web framework
