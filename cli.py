"""
Command-line interface for PGA One and Done Optimizer.
Built with Click and Rich for beautiful terminal output.
"""

import sys
import logging
from datetime import date
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.layout import Layout
from rich.live import Live
from rich import box

from .config import get_config, get_schedule, get_next_tournament, get_majors
from .database import Database
from .api import DataGolfAPI
from .scraper import Scraper
from .simulator import Simulator
from .strategy import Strategy
from .models import Tier

console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s")


@click.group()
@click.version_option(version="1.0.0", prog_name="PGA One and Done Optimizer")
def cli():
    """PGA One and Done Optimizer - Win your fantasy golf league!"""
    pass


@cli.command()
def setup():
    """First-time setup wizard."""
    console.print(Panel.fit(
        "[bold green]PGA One and Done Optimizer Setup[/]",
        subtitle="Let's get you configured!"
    ))

    config = get_config()

    # Check for API key
    if not config.datagolf_api_key:
        console.print("\n[yellow]No Data Golf API key found.[/]")
        api_key = click.prompt("Enter your Data Golf API key", default="", show_default=False)
        if api_key:
            config.datagolf_api_key = api_key

    # Risk level
    console.print("\n[cyan]Risk Level (1-10):[/]")
    console.print("  1-3: Conservative (consistent picks)")
    console.print("  4-6: Balanced (default)")
    console.print("  7-10: Aggressive (high upside)")
    risk = click.prompt("Your risk level", default=5, type=int)
    config.risk_level = max(1, min(10, risk))

    # Save config
    config.save_to_env()
    console.print("\n[green]Configuration saved![/]")

    # Initialize database
    db = Database()
    console.print("[green]Database initialized![/]")

    # Sync tournaments
    from .config import SCHEDULE_2026
    for t in SCHEDULE_2026:
        db.save_tournament(t)
    console.print(f"[green]Loaded {len(SCHEDULE_2026)} tournaments![/]")

    # Try to sync golfers if API key available
    if config.datagolf_api_key:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Syncing golfer data from API...", total=None)
            api = DataGolfAPI()
            count = api.sync_golfers_to_db()
            progress.update(task, completed=True)
        console.print(f"[green]Synced {count} golfers from Data Golf API![/]")

    console.print("\n[bold green]Setup complete! Run 'pga-oad recommend' to get picks.[/]")


@cli.command()
@click.option("--force", is_flag=True, help="Force refresh (ignore cache)")
def update(force: bool):
    """Fetch latest data from site and API."""
    console.print(Panel.fit("[bold cyan]Updating Data[/]"))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Update from API
        task1 = progress.add_task("Fetching predictions from Data Golf...", total=None)
        api = DataGolfAPI()
        preds = api.get_pre_tournament_predictions()
        api.sync_golfers_to_db()
        progress.update(task1, completed=True, description=f"[green]Got {len(preds)} predictions[/]")

        # Update from scraper
        task2 = progress.add_task("Scraping buzzfantasygolf.com...", total=None)
        try:
            with Scraper(headless=True) as scraper:
                results = scraper.refresh_all_data()
            progress.update(task2, completed=True, description=f"[green]Scraped {results.get('standings', 0)} standings[/]")
        except Exception as e:
            progress.update(task2, completed=True, description=f"[yellow]Scraper: {str(e)[:50]}[/]")

    console.print("\n[green]Update complete![/]")


@cli.command()
@click.option("--top", "-n", default=10, help="Number of recommendations")
@click.option("--tournament", "-t", default=None, help="Tournament name (default: next)")
def recommend(top: int, tournament: str):
    """Get top picks for upcoming tournament with backup choices."""
    strategy = Strategy()
    db = Database()

    # Get tournament
    if tournament:
        from .config import get_tournament_by_name
        t = get_tournament_by_name(tournament)
    else:
        t = get_next_tournament()

    if not t:
        console.print("[red]No upcoming tournament found![/]")
        return

    # Tournament info panel
    tier_colors = {Tier.TIER_1: "green", Tier.TIER_2: "yellow", Tier.TIER_3: "white"}
    tier_color = tier_colors.get(t.tier, "white")

    console.print(Panel(
        f"[bold]{t.name}[/]\n"
        f"Date: {t.date.strftime('%B %d, %Y')}\n"
        f"Course: {t.course}\n"
        f"Purse: [green]${t.purse:,}[/] | Winner: [green]${t.winner_share:,}[/]\n"
        f"Tier: [{tier_color}]{t.tier.name}[/] {'| MAJOR' if t.is_major else ''}"
        f"{'| SIGNATURE' if t.is_signature else ''}{'| PLAYOFF' if t.is_playoff else ''}",
        title="Next Tournament",
        border_style="cyan"
    ))

    # Get recommendations
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running simulations...", total=None)
        recs = strategy.get_recommendations(t, top_n=top)
        progress.update(task, completed=True)

    if not recs:
        console.print("[yellow]No recommendations available. Run 'pga-oad update' first.[/]")
        return

    # Main recommendations table
    table = Table(
        title=f"Top {top} Picks",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    table.add_column("Rank", style="bold", width=4)
    table.add_column("Golfer", style="white", width=22)
    table.add_column("OWGR", justify="center", width=5)
    table.add_column("Expected Value", justify="right", style="green", width=12)
    table.add_column("Win %", justify="right", width=7)
    table.add_column("Top-10 %", justify="right", width=8)
    table.add_column("Cut %", justify="right", width=6)
    table.add_column("Hedge", justify="right", width=6)
    table.add_column("Confidence", justify="center", width=10)

    for i, rec in enumerate(recs, 1):
        # Confidence bar
        conf_bars = int(rec.confidence * 5)
        conf_display = "[green]" + ("*" * conf_bars) + "[/]" + ("*" * (5 - conf_bars))

        # Win/Top-10 from simulation
        sim = db.get_simulation(rec.golfer.name, t.name)
        win_pct = f"{sim.win_rate*100:.1f}%" if sim else "N/A"
        top10_pct = f"{sim.top_10_rate*100:.1f}%" if sim else "N/A"
        cut_pct = f"{sim.cut_rate*100:.0f}%" if sim else "N/A"

        # Hedge bonus display
        hedge_display = f"+{rec.hedge_bonus*100:.0f}%" if rec.hedge_bonus > 0 else "-"

        # Style for top pick
        style = "bold green" if i == 1 else ("yellow" if i <= 3 else None)

        table.add_row(
            f"#{i}",
            rec.golfer.name,
            str(rec.golfer.owgr),
            f"${rec.expected_value:,.0f}",
            win_pct,
            top10_pct,
            cut_pct,
            hedge_display,
            conf_display,
            style=style
        )

    console.print(table)

    # Detailed analysis for top pick
    if recs:
        top_rec = recs[0]
        console.print(Panel(
            f"[bold green]{top_rec.golfer.name}[/] is the recommended pick.\n\n"
            f"[cyan]Analysis:[/] {top_rec.reasoning}\n\n"
            f"[cyan]Regret Risk:[/] {'LOW' if top_rec.regret_risk < 0.2 else 'MEDIUM' if top_rec.regret_risk < 0.4 else 'HIGH'}\n"
            f"[cyan]Upside (90th percentile):[/] ${db.get_simulation(top_rec.golfer.name, t.name).percentile_90:,.0f}" if db.get_simulation(top_rec.golfer.name, t.name) else "",
            title="TOP PICK ANALYSIS",
            border_style="green"
        ))

    # Backup picks
    if len(recs) > 1:
        console.print("\n[bold cyan]Backup Choices:[/]")
        for i, rec in enumerate(recs[1:4], 2):
            console.print(f"  {i}. [white]{rec.golfer.name}[/] - EV: ${rec.expected_value:,.0f} ({rec.reasoning.split('|')[0].strip()})")


@cli.command()
def standings():
    """View current league standings."""
    db = Database()
    standings = db.get_latest_standings()

    if not standings:
        console.print("[yellow]No standings data. Run 'pga-oad update' to fetch.[/]")
        return

    config = get_config()
    my_username = config.site_username.lower()

    table = Table(
        title="League Standings",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    table.add_column("Rank", justify="center", width=5)
    table.add_column("Player", width=25)
    table.add_column("Earnings", justify="right", style="green", width=12)
    table.add_column("Cuts", justify="center", width=5)

    for s in standings[:20]:
        style = "bold yellow" if s.username.lower() == my_username else None
        marker = " *" if s.username.lower() == my_username else ""
        table.add_row(
            str(s.rank),
            s.player_name + marker,
            f"${s.total_earnings:,}",
            str(s.cuts_made),
            style=style
        )

    console.print(table)

    # Show user's position
    my_standing = db.get_my_standing(config.site_username)
    if my_standing:
        console.print(f"\n[cyan]Your Position:[/] Rank #{my_standing.rank} with ${my_standing.total_earnings:,}")


@cli.command()
def schedule():
    """View 2026 tournament schedule."""
    schedule = get_schedule()
    today = date.today()

    table = Table(
        title="2026 PGA Tour Schedule",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    table.add_column("Date", width=12)
    table.add_column("Tournament", width=35)
    table.add_column("Purse", justify="right", style="green", width=12)
    table.add_column("Tier", justify="center", width=8)
    table.add_column("Type", justify="center", width=10)

    for t in schedule:
        # Skip past tournaments unless within last 2 weeks
        if t.date < today:
            continue

        tier_colors = {Tier.TIER_1: "green", Tier.TIER_2: "yellow", Tier.TIER_3: "white"}
        tier_style = tier_colors.get(t.tier, "white")

        event_type = []
        if t.is_major:
            event_type.append("MAJOR")
        if t.is_signature:
            event_type.append("SIG")
        if t.is_playoff:
            event_type.append("PLAYOFF")
        if t.is_opposite_field:
            event_type.append("OPP")

        table.add_row(
            t.date.strftime("%b %d"),
            t.name,
            f"${t.purse:,}",
            f"[{tier_style}]Tier {t.tier.value}[/]",
            " ".join(event_type) or "-"
        )

    console.print(table)

    # Summary
    total_purse = sum(t.purse for t in schedule)
    majors = [t for t in schedule if t.is_major]
    console.print(f"\n[cyan]Total Season Purse:[/] ${total_purse:,}")
    console.print(f"[cyan]Majors:[/] {', '.join(m.name for m in majors)}")


@cli.command()
def picks():
    """View your pick history."""
    db = Database()
    picks = db.get_all_picks()

    if not picks:
        console.print("[yellow]No picks recorded yet.[/]")
        return

    table = Table(
        title="Your Pick History",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan"
    )
    table.add_column("Tournament", width=30)
    table.add_column("Golfer", width=22)
    table.add_column("Position", justify="center", width=8)
    table.add_column("Earnings", justify="right", style="green", width=12)
    table.add_column("Cut", justify="center", width=5)

    for p in picks:
        pos_display = str(p.position) if p.position else "MC"
        cut_display = "[green]Y[/]" if p.made_cut else "[red]N[/]"

        table.add_row(
            p.tournament_name,
            p.golfer_name,
            pos_display,
            f"${p.earnings:,}",
            cut_display
        )

    console.print(table)

    # Summary
    total = db.get_total_earnings()
    cuts = db.get_cuts_made_count()
    console.print(f"\n[cyan]Total Earnings:[/] ${total:,}")
    console.print(f"[cyan]Cuts Made:[/] {cuts}/{len(picks)}")


@cli.command()
@click.argument("golfer")
@click.option("--tournament", "-t", default=None, help="Tournament name")
def whatif(golfer: str, tournament: str):
    """Run what-if analysis for a specific pick."""
    strategy = Strategy()
    result = strategy.what_if_pick(golfer, tournament)

    if "error" in result:
        console.print(f"[red]{result['error']}[/]")
        return

    console.print(Panel(
        f"[bold]{result['golfer']}[/] at [cyan]{result['tournament']}[/]\n\n"
        f"Purse: ${result['purse']:,}\n\n"
        f"[bold cyan]Simulation Results:[/]\n"
        f"  Expected Value: [green]${result['expected_value']:,.0f}[/]\n"
        f"  Median Value: ${result['median_value']:,.0f}\n"
        f"  Win Probability: {result['win_probability']*100:.2f}%\n"
        f"  Top-10 Probability: {result['top_10_probability']*100:.1f}%\n"
        f"  Cut Probability: {result['cut_probability']*100:.0f}%\n\n"
        f"[bold cyan]Range:[/]\n"
        f"  90th Percentile (Upside): [green]${result['upside_90th']:,.0f}[/]\n"
        f"  10th Percentile (Downside): ${result['downside_10th']:,.0f}\n\n"
        f"[bold cyan]Regret Analysis:[/]\n"
        f"  Regret Risk: ${result['regret_analysis']['regret_risk']:,.0f}\n"
        f"  Best Alternative: {result['regret_analysis'].get('best_alternative', 'N/A')}\n\n"
        f"[bold]Recommendation: [{('green' if 'PICK' in result['recommendation'] else 'yellow')}]{result['recommendation']}[/]",
        title="What-If Analysis",
        border_style="cyan"
    ))


@cli.command()
def opponents():
    """Analyze opponent usage patterns."""
    strategy = Strategy()
    analysis = strategy.analyze_opponent_patterns()

    if "message" in analysis and not analysis.get("patterns"):
        console.print(f"[yellow]{analysis['message']}[/]")
        return

    console.print(Panel.fit("[bold cyan]Opponent Strategy Analysis[/]"))

    for pattern in analysis.get("patterns", []):
        console.print(
            f"\n[bold]{pattern['type'].replace('_', ' ').title()}[/] "
            f"({pattern['count']} players, {pattern['pct']:.0f}%)\n"
            f"  {pattern['description']}\n"
            f"  Avg Elite picks: {pattern['elite_avg']:.1f}, Avg Mid-tier: {pattern['mid_avg']:.1f}"
        )

    if analysis.get("recommendation"):
        console.print(Panel(
            f"[bold green]{analysis['recommendation']}[/]",
            title="Counter-Strategy",
            border_style="green"
        ))

    # Show most/least used golfers
    db = Database()
    usage = db.get_all_golfer_usage()
    if usage:
        sorted_usage = sorted(usage.items(), key=lambda x: x[1], reverse=True)
        console.print("\n[bold cyan]Most Used Golfers:[/]")
        for name, count in sorted_usage[:5]:
            console.print(f"  {name}: {count} opponents")

        console.print("\n[bold cyan]Underused (Hedge Opportunity):[/]")
        for name, count in sorted_usage[-5:]:
            console.print(f"  {name}: {count} opponents")


@cli.command()
def plan():
    """Generate season-long pick plan."""
    strategy = Strategy()
    plan = strategy.get_season_plan()

    console.print(Panel(
        f"Strategy: {plan['strategy']}\n"
        f"Risk Level: {plan['risk_level']}/10\n"
        f"Remaining Tournaments: {plan['remaining_tournaments']}\n"
        f"Available Elites: {plan['available_elites']}\n"
        f"Available Mid-Tier: {plan['available_mid_tier']}",
        title="Season Plan",
        border_style="cyan"
    ))

    # Elite reservations
    if plan["elite_reservation"]:
        console.print("\n[bold green]Elite Reservations (Majors):[/]")
        for r in plan["elite_reservation"]:
            console.print(f"  {r['tournament']}: {r['golfer']}")

    # Signature picks
    if plan["recommendations"]["signatures"]:
        console.print("\n[bold yellow]Signature Event Picks:[/]")
        for p in plan["recommendations"]["signatures"][:5]:
            console.print(f"  {p['tournament']}: {p['golfer']} [{p['tier']}]")

    # Regular picks
    if plan["recommendations"]["regular"]:
        console.print("\n[bold white]Regular Event Picks:[/]")
        for p in plan["recommendations"]["regular"][:10]:
            console.print(f"  {p['tournament']}: {p['golfer']} [{p['tier']}]")


@cli.command()
@click.argument("golfer")
@click.option("--tournament", "-t", default=None)
@click.option("--simulations", "-n", default=50000, type=int)
def simulate(golfer: str, tournament: str, simulations: int):
    """Run Monte Carlo simulation for a golfer."""
    from .config import get_tournament_by_name

    db = Database()
    sim = Simulator(n_simulations=simulations)

    # Get golfer
    g = db.get_golfer(golfer)
    if not g:
        console.print(f"[red]Golfer '{golfer}' not found. Run 'pga-oad update' first.[/]")
        return

    # Get tournament
    if tournament:
        t = get_tournament_by_name(tournament)
    else:
        t = get_next_tournament()

    if not t:
        console.print("[red]No tournament found.[/]")
        return

    console.print(f"\n[cyan]Running {simulations:,} simulations for {g.name} at {t.name}...[/]")

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Simulating...", total=None)
        result = sim.simulate_tournament(g, t, simulations)
        progress.update(task, completed=True)

    # Display results
    table = Table(title="Simulation Results", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right", style="green")

    table.add_row("Expected Value", f"${result.mean_earnings:,.0f}")
    table.add_row("Median Value", f"${result.median_earnings:,.0f}")
    table.add_row("Std Deviation", f"${result.std_earnings:,.0f}")
    table.add_row("10th Percentile", f"${result.percentile_10:,.0f}")
    table.add_row("25th Percentile", f"${result.percentile_25:,.0f}")
    table.add_row("75th Percentile", f"${result.percentile_75:,.0f}")
    table.add_row("90th Percentile", f"${result.percentile_90:,.0f}")
    table.add_row("Win Rate", f"{result.win_rate*100:.2f}%")
    table.add_row("Top-10 Rate", f"{result.top_10_rate*100:.1f}%")
    table.add_row("Cut Rate", f"{result.cut_rate*100:.0f}%")

    console.print(table)


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
