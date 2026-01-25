"""
Streamlit Web Application for PGA One and Done Optimizer.
A fun, interactive UI accessible remotely.

Run with: streamlit run web_app.py --server.port 8501 --server.address 0.0.0.0
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import get_config, get_schedule, get_next_tournament, get_majors, get_tournament_by_name, get_no_cut_events
from database import Database
from api import DataGolfAPI
from simulator import Simulator
from strategy import Strategy
from models import Tier, CutRule
from learner import LearningEngine

# Page config
st.set_page_config(
    page_title="PGA One and Done Optimizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling (compatible with dark/light themes)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1a472a 0%, #2d5a3d 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .pick-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 0.5rem 0;
    }
    .recommendation-1 { border-left: 5px solid #FFD700; }
    .recommendation-2 { border-left: 5px solid #C0C0C0; }
    .recommendation-3 { border-left: 5px solid #CD7F32; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'db' not in st.session_state:
        st.session_state.db = Database()
    if 'strategy' not in st.session_state:
        st.session_state.strategy = Strategy()
    if 'simulator' not in st.session_state:
        st.session_state.simulator = Simulator()
    if 'api' not in st.session_state:
        st.session_state.api = DataGolfAPI()
    if 'learner' not in st.session_state:
        st.session_state.learner = LearningEngine(st.session_state.db)

    # Check if data sync is needed (only on initial load)
    if 'data_sync_checked' not in st.session_state:
        st.session_state.data_sync_checked = True
        db = st.session_state.db
        golfer_count = db.get_golfer_count()
        valid_owgr_count = db.get_valid_owgr_count()

        # Data sync is needed if no golfers or all have default OWGR 999
        st.session_state.data_sync_needed = (golfer_count == 0 or valid_owgr_count == 0)


def main():
    """Main application."""
    init_session_state()

    # Header
    st.markdown('<div class="main-header">PGA One and Done Optimizer</div>', unsafe_allow_html=True)

    # Data sync warning banner
    if st.session_state.get('data_sync_needed', False):
        st.warning("**No golfer data found.** Your database appears to be empty or missing valid data.")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("Sync Data Now", type="primary"):
                with st.spinner("Syncing golfer data from Data Golf API..."):
                    try:
                        api = st.session_state.api
                        count = api.sync_golfers_to_db()
                        preds = api.get_pre_tournament_predictions()
                        st.session_state.data_sync_needed = False
                        st.success(f"Synced {count} golfers and {len(preds)} predictions!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Sync failed: {e}")
        with col2:
            st.caption("This will fetch golfer data, rankings, and predictions from the Data Golf API.")
        st.divider()

    # Sidebar navigation
    st.sidebar.image("https://www.pgatour.com/content/dam/pgatour/logos/pga-tour-logo.svg", width=150)
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Select Page",
        ["Dashboard", "Pick Recommendations", "Betting & Odds", "Tournament Schedule",
         "What-If Analysis", "Season Planner", "Season Simulation", "Multi-Entry",
         "Learning Insights", "League Standings", "Settings"]
    )

    if page == "Dashboard":
        show_dashboard()
    elif page == "Pick Recommendations":
        show_recommendations()
    elif page == "Betting & Odds":
        show_betting_odds()
    elif page == "Tournament Schedule":
        show_schedule()
    elif page == "What-If Analysis":
        show_whatif()
    elif page == "Season Planner":
        show_planner()
    elif page == "Season Simulation":
        show_season_simulation()
    elif page == "Multi-Entry":
        show_multi_entry()
    elif page == "Learning Insights":
        show_learning_insights()
    elif page == "League Standings":
        show_standings()
    elif page == "Settings":
        show_settings()


def show_dashboard():
    """Main dashboard with overview."""
    st.title("Dashboard")

    # Next tournament info
    next_t = get_next_tournament()

    col1, col2, col3 = st.columns(3)

    with col1:
        if next_t:
            st.metric("Next Tournament", next_t.name)
            st.caption(f"{next_t.date.strftime('%B %d, %Y')}")

    with col2:
        if next_t:
            st.metric("Purse", f"${next_t.purse:,}")
            st.caption(f"Winner: ${next_t.winner_share:,}")

    with col3:
        db = st.session_state.db
        total_earnings = db.get_total_earnings()
        st.metric("Your Earnings", f"${total_earnings:,}")
        st.caption(f"{db.get_picks_count()} picks made")

    # Weather forecast for tournament
    if next_t:
        from api import get_weather_api
        from config import get_course_profile

        profile = get_course_profile(next_t.course)
        if profile and profile.latitude != 0:
            with st.expander("🌤️ Tournament Weather Forecast", expanded=False):
                weather_api = get_weather_api()
                weather = weather_api.get_tournament_weather(profile.latitude, profile.longitude)

                if weather and weather.get("forecasts"):
                    forecasts = weather["forecasts"][:5]  # Next 5 days

                    weather_cols = st.columns(len(forecasts))
                    for i, forecast in enumerate(forecasts):
                        with weather_cols[i]:
                            st.markdown(f"**{forecast['date']}**")
                            st.write(f"🌡️ {forecast['temp_high_c']:.0f}°C")
                            st.write(f"💨 {forecast['wind_max_mph']:.0f} mph")
                            if forecast['precipitation_probability'] > 30:
                                st.write(f"🌧️ {forecast['precipitation_probability']}%")

                            # Scoring impact
                            impact = forecast['scoring_impact']
                            if impact > 0.3:
                                st.error(f"+{impact:.1f} strokes")
                            elif impact > 0.1:
                                st.warning(f"+{impact:.1f} strokes")
                            else:
                                st.success("Normal")

                    # Show conditions summary
                    windy_days = sum(1 for f in forecasts if f['wind_max_mph'] > 15)
                    if windy_days >= 2:
                        st.warning(f"⚠️ Windy conditions expected ({windy_days} days with 15+ mph winds) - favors accurate drivers")
                else:
                    st.info("Weather data not available")

    st.divider()

    # Quick recommendations
    st.subheader("Quick Pick Preview")

    if next_t:
        strategy = st.session_state.strategy

        try:
            with st.spinner("Running simulations..."):
                recs = strategy.get_recommendations(next_t, top_n=5)
        except Exception as e:
            st.error(f"Failed to generate recommendations: {e}")
            recs = []

        if recs:
            # Top pick highlight
            top = recs[0]
            st.success(f"**TOP PICK: {top.golfer.name}** - Expected Value: ${top.expected_value:,.0f}")

            # Quick comparison chart
            fig = go.Figure()

            names = [r.golfer.name for r in recs]
            evs = [r.expected_value for r in recs]

            fig.add_trace(go.Bar(
                x=names,
                y=evs,
                marker_color=['#FFD700', '#C0C0C0', '#CD7F32', '#4CAF50', '#2196F3'],
                text=[f"${ev:,.0f}" for ev in evs],
                textposition='outside'
            ))

            fig.update_layout(
                title="Top 5 Picks by Expected Value",
                xaxis_title="Golfer",
                yaxis_title="Expected Value ($)",
                showlegend=False,
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No recommendations available yet. Click 'Sync Data Now' above or go to Settings to update data.")

    # Season progress
    st.subheader("Season Progress")

    schedule = get_schedule()
    today = date.today()
    total = len(schedule)
    completed = len([t for t in schedule if t.date < today])

    progress = completed / total if total > 0 else 0
    st.progress(progress, text=f"{completed}/{total} tournaments completed ({progress*100:.0f}%)")

    # Win Target Tracking (2025 Blueprint: 4-6 wins, double top-5s)
    st.subheader("Win Target Tracking")
    st.caption("2025 Blueprint: 4-6 wins with close to double Top 5 finishes")

    strategy = st.session_state.strategy
    win_stats = strategy.get_season_win_targets()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        target_wins = 5
        win_color = "normal" if win_stats["wins"] >= (progress * target_wins) else "inverse"
        st.metric("Wins", win_stats["wins"], delta=f"Target: {target_wins}", delta_color=win_color)
    with col2:
        st.metric("Top 5s", win_stats["top_5s"], delta=f"Need: {win_stats['wins'] * 2}")
    with col3:
        st.metric("Top 10s", win_stats["top_10s"])
    with col4:
        st.metric("Earnings", f"${win_stats['total_earnings']:,}")

    # Win progress analysis
    if win_stats["analysis"]:
        if "AHEAD" in win_stats["analysis"]:
            st.success(win_stats["analysis"])
        elif "ON TRACK" in win_stats["analysis"]:
            st.info(win_stats["analysis"])
        elif "BEHIND" in win_stats["analysis"]:
            st.warning(win_stats["analysis"])
        else:
            st.info(win_stats["analysis"])

    # Standings-based strategy
    st.subheader("Strategy Mode")
    standings_mode, standings_mult, standings_explanation = strategy.get_standings_strategy()

    if standings_mode == "protect":
        st.success(f"**{standings_mode.upper()}**: {standings_explanation}")
    elif standings_mode in ("aggressive", "desperation"):
        st.warning(f"**{standings_mode.upper()}**: {standings_explanation}")
    else:
        st.info(f"**{standings_mode.upper()}**: {standings_explanation}")


def show_recommendations():
    """Detailed pick recommendations with backups."""
    st.title("Pick Recommendations")

    # Tournament selector
    schedule = get_schedule()
    upcoming = [t for t in schedule if t.date >= date.today()]
    tournament_names = [t.name for t in upcoming]

    selected = st.selectbox("Select Tournament", tournament_names)
    tournament = get_tournament_by_name(selected) if selected else get_next_tournament()

    if not tournament:
        st.warning("No tournament selected")
        return

    # Tournament info card
    tier_emoji = {Tier.TIER_1: "", Tier.TIER_2: "", Tier.TIER_3: ""}
    cut_info = "NO CUT" if not tournament.has_cut else f"Cut: {tournament.cut_rule.value.replace('_', ' ').title()}"

    st.info(f"""
    **{tournament.name}** {tier_emoji.get(tournament.tier, '')}

    **Date:** {tournament.date.strftime('%B %d, %Y')} |
    **Purse:** ${tournament.purse:,} |
    **Winner:** ${tournament.winner_share:,} |
    **Tier:** {tournament.tier.name} |
    **{cut_info}**
    {"| **MAJOR**" if tournament.is_major else ""}{"| **SIGNATURE**" if tournament.is_signature else ""}
    """)

    # No-cut event advisory
    if not tournament.has_cut:
        st.success(f"This is a NO-CUT event. All {tournament.field_size} players are guaranteed payment (min ${tournament.min_payout:,}). Higher-variance picks are favored.")

    # Opposite-field event advisory (Phase 1.3)
    if tournament.is_opposite_field:
        st.info("OPPOSITE FIELD EVENT: Weaker field - mid-tier golfers (OWGR 30-60) get a 15% EV boost as they can compete for wins.")

    # Number of recommendations
    num_recs = st.slider("Number of recommendations", 5, 20, 10)

    # Generate recommendations
    if st.button("Generate Recommendations", type="primary"):
        strategy = st.session_state.strategy

        with st.spinner("Running 50,000 Monte Carlo simulations per golfer..."):
            recs = strategy.get_recommendations(tournament, top_n=num_recs)

        # Display field strength indicator (Phase 1.1)
        if recs:
            first_rec = recs[0]
            field_strength = first_rec.field_strength
            if field_strength == "WEAK":
                st.warning(f"**WEAK FIELD** - Mid-tier golfers boosted +10-15%")
            elif field_strength == "STRONG":
                st.info(f"**STRONG FIELD** - Elite competition, non-elites penalized")
            else:
                st.caption(f"**MODERATE FIELD** - Standard competition")

        if not recs:
            st.error("No recommendations available. Please update data first.")

            # Troubleshooting expander
            with st.expander("Troubleshooting Info"):
                db = st.session_state.db
                api = st.session_state.api

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Golfers in DB", db.get_golfer_count())
                with col2:
                    st.metric("Valid OWGR Count", db.get_valid_owgr_count())
                with col3:
                    st.metric("Golfers Used", len(db.get_used_golfers()))

                if api.last_error:
                    st.error(f"Last API Error: {api.last_error}")

                st.info("Try: Go to Settings → Click 'Update from API' or 'Clear All Cache' first.")
            return

        st.success(f"Generated {len(recs)} recommendations!")

        # Store in session for persistence
        st.session_state.last_recs = recs

        # Record predictions for learning (only top 20)
        try:
            learner = st.session_state.learner
            predictions_data = []
            for rec in recs[:20]:
                predictions_data.append({
                    "golfer_name": rec.golfer.name,
                    "win_prob": rec.golfer.win_probability or 0,
                    "top_10_prob": rec.golfer.top_10_probability or 0,
                    "expected_value": rec.expected_value,
                    "course_fit": rec.course_fit_sg if hasattr(rec, 'course_fit_sg') else 0,
                    "strategic_score": rec.expected_value,
                    "was_save_warning": hasattr(rec, 'relative_value') and rec.relative_value < 0.7
                })
            learner.record_recommendations(
                tournament_name=tournament.name,
                tournament_date=tournament.date,
                purse=tournament.purse,
                recommendations=predictions_data
            )
            st.caption("Predictions recorded for learning analysis")
        except Exception as e:
            st.caption(f"Note: Could not record predictions - {e}")

    # Display recommendations
    if 'last_recs' in st.session_state:
        recs = st.session_state.last_recs
        db = st.session_state.db

        # Tabs for different views
        tab1, tab2, tab3 = st.tabs(["Detailed View", "Comparison Chart", "Data Table"])

        with tab1:
            st.subheader("Detailed Recommendations")

            for i, rec in enumerate(recs, 1):
                sim = db.get_simulation(rec.golfer.name, tournament.name)

                # Calculate probabilities for display
                if sim:
                    win_pct = sim.win_rate * 100
                    top10_pct = sim.top_10_rate * 100
                else:
                    win_pct = rec.golfer.win_probability * 100 if rec.golfer.win_probability else 0
                    top10_pct = rec.golfer.top_10_probability * 100 if rec.golfer.top_10_probability else 0

                # Medal colors for top 3
                if i == 1:
                    medal = ""
                    color = "#FFD700"
                elif i == 2:
                    medal = ""
                    color = "#C0C0C0"
                elif i == 3:
                    medal = ""
                    color = "#CD7F32"
                else:
                    medal = f"#{i}"
                    color = "#4CAF50"

                # Show OWGR warning and cut risk in the expander title if applicable
                owgr_flag = " OWGR RISK" if rec.owgr_warning else ""
                cut_flag = " CUT RISK" if rec.cut_warning else ""
                with st.expander(f"{medal} **{rec.golfer.name}** | Win: {win_pct:.1f}% | EV: ${rec.expected_value:,.0f}{owgr_flag}{cut_flag}", expanded=(i <= 3)):
                    # Cut probability warning (Phase 1.2)
                    if rec.cut_warning:
                        st.error(f"CUT RISK: This golfer has less than 80% probability to make the cut. EV has been penalized.")

                    # OWGR Warning banner
                    if rec.owgr_warning:
                        st.error(f"OWGR Warning: Rank {rec.golfer.owgr} is outside top 65. Historical winners avoid these picks.")

                    # Opportunity cost warning for elite picks at non-major events
                    is_elite = rec.golfer.owgr <= 20
                    if is_elite and not tournament.is_major and not tournament.is_signature:
                        # Estimate opportunity cost - elite picks are worth more at majors
                        majors = get_majors()
                        remaining_majors = [m for m in majors if m.date >= date.today()]
                        if remaining_majors:
                            avg_major_purse = sum(m.winner_share for m in remaining_majors) / len(remaining_majors)
                            # Estimate major EV based on win probability
                            major_ev = win_pct / 100 * avg_major_purse * 0.5  # Rough estimate
                            if major_ev > rec.expected_value * 1.5:
                                st.warning(f"Opportunity Cost: Elite pick (OWGR #{rec.golfer.owgr}) could be worth ${major_ev:,.0f} at a major")

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Expected Value", f"${rec.expected_value:,.0f}")
                        owgr_color = "inverse" if rec.owgr_warning else "normal"
                        st.metric("OWGR", rec.golfer.owgr)

                    with col2:
                        st.metric("Win %", f"{win_pct:.2f}%")
                        st.metric("Top-10 %", f"{top10_pct:.1f}%")

                    with col3:
                        if sim:
                            cut_pct = sim.cut_rate * 100
                        else:
                            cut_pct = rec.golfer.make_cut_probability * 100 if rec.golfer.make_cut_probability else 85
                        if tournament.has_cut:
                            st.metric("Cut %", f"{cut_pct:.0f}%")
                        else:
                            st.metric("Guaranteed", "NO CUT")
                        st.metric("Confidence", f"{rec.confidence*100:.0f}%")

                    with col4:
                        if sim:
                            st.metric("Upside (90th)", f"${sim.percentile_90:,.0f}")
                            st.metric("Downside (10th)", f"${sim.percentile_10 if tournament.has_cut else tournament.min_payout:,.0f}")
                        else:
                            st.metric("Upside (90th)", "N/A")
                            st.metric("Downside (10th)", "N/A")

                    # Course Fit indicator
                    if rec.course_fit_sg != 0:
                        fit_color = "green" if rec.course_fit_sg > 0 else "red"
                        fit_sign = "+" if rec.course_fit_sg > 0 else ""
                        st.info(f"Course Fit: {fit_sign}{rec.course_fit_sg:.2f} SG/round (Data Golf adjustment)")

                    # RELATIVE VALUE indicator (opportunity cost)
                    if hasattr(rec, 'relative_value'):
                        if rec.relative_value >= 1.0:
                            st.success(f"✓ OPTIMAL TIMING: This is {rec.golfer.name}'s best remaining opportunity ({rec.relative_value:.0%} of max EV)")
                        elif rec.relative_value >= 0.85:
                            st.info(f"Good timing: {rec.relative_value:.0%} of max EV" + (f" (Better at {rec.best_future_event})" if rec.best_future_event else ""))
                        elif rec.relative_value >= 0.7:
                            st.warning(f"⚠️ Consider saving: Only {rec.relative_value:.0%} of max EV here. Better opportunity at {rec.best_future_event}")
                        else:
                            st.error(f"🚫 SAVE THIS GOLFER: Only {rec.relative_value:.0%} of max EV. Much better at {rec.best_future_event}")

                    # Hedge bonus indicator
                    if rec.hedge_bonus > 0:
                        st.success(f"Hedge Bonus: +{rec.hedge_bonus*100:.0f}% (underused by opponents)")

                    # Risk indicator
                    if rec.regret_risk > 0.3:
                        st.warning("Higher regret risk - consider alternatives")
                    elif rec.regret_risk < 0.1:
                        st.success("Low regret risk - solid choice!")

                    # Expandable reasoning section
                    with st.expander("Why this pick?"):
                        st.write(rec.reasoning)

        with tab2:
            st.subheader("Visual Comparison")

            # Expected Value comparison
            fig1 = go.Figure()

            for i, rec in enumerate(recs[:10]):
                sim = db.get_simulation(rec.golfer.name, tournament.name)
                if sim:
                    fig1.add_trace(go.Box(
                        y=[sim.percentile_10, sim.percentile_25, sim.median_earnings,
                           sim.percentile_75, sim.percentile_90],
                        name=rec.golfer.name[:15],
                        boxpoints=False
                    ))

            fig1.update_layout(
                title="Earnings Distribution by Golfer",
                yaxis_title="Projected Earnings ($)",
                showlegend=False,
                height=500
            )

            st.plotly_chart(fig1, use_container_width=True)

            # Win probability bubble chart
            fig2 = go.Figure()

            for rec in recs[:15]:
                sim = db.get_simulation(rec.golfer.name, tournament.name)
                if sim:
                    fig2.add_trace(go.Scatter(
                        x=[sim.win_rate * 100],
                        y=[rec.expected_value],
                        mode='markers+text',
                        marker=dict(size=sim.cut_rate * 50, opacity=0.6),
                        text=[rec.golfer.name],
                        textposition="top center",
                        name=rec.golfer.name
                    ))

            fig2.update_layout(
                title="Win % vs Expected Value (bubble size = cut %)",
                xaxis_title="Win Probability (%)",
                yaxis_title="Expected Value ($)",
                showlegend=False,
                height=500
            )

            st.plotly_chart(fig2, use_container_width=True)

        with tab3:
            st.subheader("Full Data Table")

            data = []
            for i, rec in enumerate(recs, 1):
                sim = db.get_simulation(rec.golfer.name, tournament.name)
                row = {
                    "Rank": i,
                    "Golfer": rec.golfer.name,
                    "OWGR": rec.golfer.owgr,
                    "Expected Value": rec.expected_value,
                    "Win %": sim.win_rate * 100 if sim else 0,
                    "Top-10 %": sim.top_10_rate * 100 if sim else 0,
                }
                if tournament.has_cut:
                    row["Cut %"] = sim.cut_rate * 100 if sim else 0
                else:
                    row["Guaranteed"] = "YES"
                row.update({
                    "Upside (90th)": sim.percentile_90 if sim else 0,
                    "Downside (10th)": sim.percentile_10 if sim else (tournament.min_payout if not tournament.has_cut else 0),
                    "Course Fit": f"{rec.course_fit_sg:+.2f}" if rec.course_fit_sg else "-",
                    "Hedge Bonus": f"+{rec.hedge_bonus*100:.0f}%" if rec.hedge_bonus > 0 else "-",
                    "Confidence": f"{rec.confidence*100:.0f}%",
                    "OWGR Risk": "YES" if rec.owgr_warning else "-",
                    "Cut Risk": "HIGH" if rec.cut_warning else "-",  # Phase 1.2
                    "Field": rec.field_strength if rec.field_strength else "-",  # Phase 1.1
                })
                data.append(row)

            df = pd.DataFrame(data)
            st.dataframe(
                df.style.format({
                    "Expected Value": "${:,.0f}",
                    "Win %": "{:.2f}%",
                    "Top-10 %": "{:.1f}%",
                    "Cut %": "{:.0f}%",
                    "Upside (90th)": "${:,.0f}",
                    "Downside (10th)": "${:,.0f}"
                }),
                use_container_width=True,
                height=500
            )

            # Download button
            csv = df.to_csv(index=False)
            st.download_button(
                "Download as CSV",
                csv,
                f"recommendations_{tournament.name.replace(' ', '_')}.csv",
                "text/csv"
            )


def show_betting_odds():
    """Betting odds comparison and approach skill breakdown."""
    st.title("Betting Odds & Player Analysis")

    api = st.session_state.api

    tab1, tab2, tab3 = st.tabs(["Betting Odds", "Approach Skills", "Course Fit"])

    with tab1:
        st.subheader("Betting Odds Comparison")
        st.markdown("Compare Data Golf model predictions vs. market consensus from 11 sportsbooks.")

        market = st.selectbox("Market", ["win", "top_5", "top_10", "top_20", "make_cut"])

        if st.button("Fetch Betting Odds", key="fetch_odds"):
            with st.spinner("Fetching odds from sportsbooks..."):
                odds = api.get_betting_outrights(market=market)

            if odds:
                data = []
                for golfer, books in odds.items():
                    dg_prob = books.get("datagolf_model", 0) * 100
                    consensus = books.get("consensus", 0) * 100
                    if dg_prob > 0 or consensus > 0:
                        # Value = DG model - market (positive = undervalued by market)
                        value = dg_prob - consensus
                        data.append({
                            "Golfer": golfer,
                            "DG Model %": dg_prob,
                            "Market %": consensus,
                            "Value": value,
                            "Edge": "Undervalued" if value > 1 else ("Overvalued" if value < -1 else "Fair")
                        })

                df = pd.DataFrame(data)
                if not df.empty:
                    df = df.sort_values("Value", ascending=False)

                    # Display without background_gradient for compatibility
                    st.dataframe(
                        df.style.format({
                            "DG Model %": "{:.2f}%",
                            "Market %": "{:.2f}%",
                            "Value": "{:+.2f}%"
                        }),
                        use_container_width=True,
                        height=500
                    )

                    # Top value picks
                    st.subheader("Top Value Picks (Model > Market)")
                    top_value = df[df["Value"] > 0].head(10)
                    if not top_value.empty:
                        for _, row in top_value.iterrows():
                            st.success(f"**{row['Golfer']}**: DG Model {row['DG Model %']:.2f}% vs Market {row['Market %']:.2f}% (+{row['Value']:.2f}% edge)")
                    else:
                        st.info("No undervalued picks found in this market.")
                else:
                    st.warning("No odds data to display.")
            else:
                st.warning("No betting odds available. Ensure API key is configured.")

        st.divider()

        # The Odds API - Additional sportsbook source
        st.subheader("🆕 The Odds API (DraftKings, FanDuel, BetMGM)")
        st.markdown("Additional odds aggregation from US sportsbooks. Requires ODDS_API_KEY (free tier: 500 requests/month).")
        st.caption("Get API key at: https://the-odds-api.com/")

        from api import get_odds_api
        odds_api = get_odds_api()

        if odds_api.is_configured():
            if st.button("Fetch The Odds API", key="fetch_theodds"):
                with st.spinner("Fetching odds from DraftKings, FanDuel, BetMGM..."):
                    golf_sports = odds_api.get_golf_sports()
                    active_sports = [s for s in golf_sports if s.get("active")]

                    if active_sports:
                        st.info(f"Active golf market: **{active_sports[0]['title']}**")
                        market_odds = odds_api.get_tournament_odds()

                        if market_odds:
                            data = []
                            for golfer, odds_data in market_odds.items():
                                data.append({
                                    "Golfer": golfer,
                                    "Consensus %": odds_data.get("consensus_prob", 0) * 100,
                                    "Best Odds": odds_data.get("best_odds", 0),
                                    "# Books": odds_data.get("num_books", 0),
                                })

                            df = pd.DataFrame(data)
                            df = df.sort_values("Consensus %", ascending=False)

                            st.dataframe(
                                df.style.format({
                                    "Consensus %": "{:.2f}%",
                                    "Best Odds": "{:+.0f}",
                                }),
                                use_container_width=True,
                                height=400
                            )

                            # Value comparison with DG model
                            st.subheader("Value vs Data Golf Model")
                            predictions = api.get_pre_tournament_predictions()
                            model_probs = {p.golfer_name: p.win_prob for p in predictions}

                            value_plays = odds_api.compare_to_model(model_probs, market_odds)
                            if value_plays:
                                for play in value_plays[:5]:
                                    edge = play["edge"] * 100
                                    if edge > 1:
                                        st.success(f"**{play['name']}**: Model {play['model_prob']*100:.1f}% vs Market {play['market_prob']*100:.1f}% = **+{edge:.1f}% edge** (Best odds: {play['best_odds']:+.0f})")
                                    elif edge > 0:
                                        st.info(f"**{play['name']}**: +{edge:.1f}% edge")
                        else:
                            st.warning("No odds returned from The Odds API")
                    else:
                        st.info("No active golf tournaments found on The Odds API")
        else:
            st.info("ODDS_API_KEY not configured. Set the environment variable to enable this feature.")

    with tab2:
        st.subheader("Approach Skill by Yardage")
        st.markdown("Player performance (strokes gained) broken down by approach distance.")

        if st.button("Load Approach Skills", key="load_approach"):
            with st.spinner("Fetching approach skill data..."):
                approach_data = api.get_approach_skill()

            if approach_data:
                data = []
                for golfer, buckets in approach_data.items():
                    data.append({
                        "Golfer": golfer,
                        "50-100 yds": buckets.sg_50_100,
                        "100-150 yds": buckets.sg_100_150,
                        "150-200 yds": buckets.sg_150_200,
                        "200+ yds": buckets.sg_200_plus,
                        "Rough >150": buckets.sg_fairway,
                        "Rough <150": buckets.sg_rough,
                    })

                df = pd.DataFrame(data)

                if not df.empty:
                    # Filter options
                    sort_by = st.selectbox("Sort by", ["50-100 yds", "100-150 yds", "150-200 yds", "200+ yds", "Rough >150", "Rough <150"])
                    df = df.sort_values(sort_by, ascending=False)

                    # Display without background_gradient to avoid matplotlib dependency issues
                    st.dataframe(
                        df.style.format({
                            "50-100 yds": "{:.3f}",
                            "100-150 yds": "{:.3f}",
                            "150-200 yds": "{:.3f}",
                            "200+ yds": "{:.3f}",
                            "Rough >150": "{:.3f}",
                            "Rough <150": "{:.3f}",
                        }),
                        use_container_width=True,
                        height=500
                    )

                    # Chart for top players
                    top_10 = df.head(10)
                    fig = go.Figure()
                    for col in ["50-100 yds", "100-150 yds", "150-200 yds", "200+ yds"]:
                        fig.add_trace(go.Bar(name=col, x=top_10["Golfer"], y=top_10[col]))
                    fig.update_layout(
                        title=f"Top 10 Players by {sort_by} - Approach Breakdown",
                        barmode='group',
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No approach skill data found.")
            else:
                st.warning("No approach skill data available.")

    with tab3:
        st.subheader("Course Fit Adjustments")

        # Show which tournament/course is being analyzed
        next_t = get_next_tournament()
        if next_t:
            st.info(f"**Analyzing:** {next_t.name} at **{next_t.course}** ({next_t.date.strftime('%B %d, %Y')})")
        st.markdown("How much a player gains/loses (SG/round) based on their skills matching the course characteristics.")

        if st.button("Load Course Fit Data", key="load_fit"):
            with st.spinner("Calculating course fit based on player skills..."):
                fit_data = api.get_course_fit_predictions()

            if fit_data:
                data = []
                for golfer, adjustment in fit_data.items():
                    if adjustment != 0:
                        data.append({
                            "Golfer": golfer,
                            "Course Fit": adjustment,
                            "Impact": "Strong Positive" if adjustment > 0.3 else ("Positive" if adjustment > 0 else ("Negative" if adjustment > -0.3 else "Strong Negative"))
                        })

                if data:
                    df = pd.DataFrame(data)
                    df = df.sort_values("Course Fit", ascending=False)

                    # Display without background_gradient for compatibility
                    st.dataframe(
                        df.style.format({
                            "Course Fit": "{:+.3f}",
                        }),
                        use_container_width=True,
                        height=500
                    )

                    # Top course fits
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Best Course Fits")
                        for _, row in df.head(5).iterrows():
                            st.success(f"**{row['Golfer']}**: +{row['Course Fit']:.3f} SG/round")

                    with col2:
                        st.subheader("Worst Course Fits")
                        for _, row in df.tail(5).iterrows():
                            st.error(f"**{row['Golfer']}**: {row['Course Fit']:.3f} SG/round")
                else:
                    st.warning("No course fit adjustments found (all players have neutral fit).")
            else:
                st.warning("No course fit data available.")

        st.divider()

        # Data Golf Player Decompositions (real course fit from their model)
        st.subheader("🆕 Data Golf Course Fit (Live API)")
        st.markdown("**Real-time** course fit predictions directly from Data Golf's model - includes baseline skill, course history, and course fit adjustments.")

        if st.button("Load Data Golf Decompositions", key="load_decomp"):
            with st.spinner("Fetching player decompositions from Data Golf API..."):
                decomps = api.get_player_decompositions()

            if decomps:
                data = []
                for golfer, d in decomps.items():
                    total_adj = d.get("course_history_adj", 0) + d.get("course_fit_adj", 0)
                    data.append({
                        "Golfer": golfer,
                        "Baseline": d.get("baseline_pred", 0),
                        "Course History": d.get("course_history_adj", 0),
                        "Course Fit": d.get("course_fit_adj", 0),
                        "Total Adj": total_adj,
                        "Final Pred": d.get("total_pred", 0),
                    })

                if data:
                    df = pd.DataFrame(data)
                    df = df.sort_values("Total Adj", ascending=False)

                    st.dataframe(
                        df.style.format({
                            "Baseline": "{:.2f}",
                            "Course History": "{:+.2f}",
                            "Course Fit": "{:+.2f}",
                            "Total Adj": "{:+.2f}",
                            "Final Pred": "{:.2f}",
                        }),
                        use_container_width=True,
                        height=400
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("📈 Best Course Fits (DG)")
                        top_fits = df.nlargest(5, "Total Adj")
                        for _, row in top_fits.iterrows():
                            adj = row["Total Adj"]
                            if adj > 0:
                                st.success(f"**{row['Golfer']}**: +{adj:.2f} (History: {row['Course History']:+.2f}, Fit: {row['Course Fit']:+.2f})")
                            else:
                                st.info(f"**{row['Golfer']}**: {adj:+.2f}")

                    with col2:
                        st.subheader("📉 Worst Course Fits (DG)")
                        bottom_fits = df.nsmallest(5, "Total Adj")
                        for _, row in bottom_fits.iterrows():
                            adj = row["Total Adj"]
                            st.error(f"**{row['Golfer']}**: {adj:+.2f} (History: {row['Course History']:+.2f}, Fit: {row['Course Fit']:+.2f})")
                else:
                    st.warning("No decomposition data returned.")
            else:
                st.warning("Could not fetch player decompositions. Check API connection.")


def show_schedule():
    """Tournament schedule view."""
    st.title("2026 PGA Tour Schedule")

    schedule = get_schedule()

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        show_past = st.checkbox("Show past tournaments", value=False)
    with col2:
        tier_filter = st.multiselect(
            "Filter by tier",
            ["Tier 1", "Tier 2", "Tier 3"],
            default=["Tier 1", "Tier 2", "Tier 3"]
        )

    # Filter schedule
    today = date.today()
    filtered = []
    for t in schedule:
        if not show_past and t.date < today:
            continue
        tier_name = f"Tier {t.tier.value}"
        if tier_name not in tier_filter:
            continue
        filtered.append(t)

    # Display as table
    data = []
    for t in filtered:
        event_type = []
        if t.is_major:
            event_type.append("MAJOR")
        if t.is_signature:
            event_type.append("SIGNATURE")
        if t.is_playoff:
            event_type.append("PLAYOFF")
        if t.is_opposite_field:
            event_type.append("OPPOSITE")

        # Cut rule display
        if not t.has_cut:
            cut_info = "NO CUT"
        elif t.cut_rule == CutRule.TOP_50_TIES:
            cut_info = "Top 50+T"
        else:
            cut_info = "Standard"

        data.append({
            "Date": t.date.strftime("%b %d, %Y"),
            "Tournament": t.name,
            "Course": t.course,
            "Purse": t.purse,
            "Winner's Share": t.winner_share,
            "Field": t.field_size,
            "Cut": cut_info,
            "Tier": t.tier.name,
            "Type": ", ".join(event_type) if event_type else "-"
        })

    df = pd.DataFrame(data)
    st.dataframe(
        df.style.format({
            "Purse": "${:,}",
            "Winner's Share": "${:,}"
        }),
        use_container_width=True,
        height=600
    )

    # Summary stats
    st.divider()
    col1, col2, col3, col4, col5 = st.columns(5)

    total_purse = sum(t.purse for t in filtered)
    majors = [t for t in filtered if t.is_major]
    tier1 = [t for t in filtered if t.tier == Tier.TIER_1]
    no_cut = [t for t in filtered if not t.has_cut]

    with col1:
        st.metric("Total Events", len(filtered))
    with col2:
        st.metric("Total Purse", f"${total_purse:,}")
    with col3:
        st.metric("Majors", len(majors))
    with col4:
        st.metric("Tier 1 Events", len(tier1))
    with col5:
        st.metric("No-Cut Events", len(no_cut))


def show_whatif():
    """What-if analysis tool."""
    st.title("What-If Analysis")

    st.markdown("""
    Compare different pick scenarios and see detailed simulation results.
    """)

    # Golfer input
    db = st.session_state.db
    golfers = db.get_all_golfers()
    golfer_names = [g.name for g in golfers]

    if not golfer_names:
        st.warning("No golfer data. Please update data first.")
        return

    col1, col2 = st.columns(2)

    with col1:
        golfer1 = st.selectbox("Primary Golfer", golfer_names, key="g1")

    with col2:
        golfer2 = st.selectbox("Compare With", [n for n in golfer_names if n != golfer1], key="g2")

    # Tournament
    schedule = get_schedule()
    upcoming = [t for t in schedule if t.date >= date.today()]
    tournament_names = [t.name for t in upcoming]
    selected_t = st.selectbox("Tournament", tournament_names)
    tournament = get_tournament_by_name(selected_t)

    if st.button("Run Comparison", type="primary"):
        simulator = st.session_state.simulator

        g1 = db.get_golfer(golfer1)
        g2 = db.get_golfer(golfer2)

        if not g1 or not g2:
            st.error("Could not find golfer data. Please sync data first in Settings.")
            return

        if not tournament:
            st.error("Could not find tournament data.")
            return

        try:
            with st.spinner("Running simulations..."):
                sim1 = simulator.simulate_tournament(g1, tournament)
                sim2 = simulator.simulate_tournament(g2, tournament)
        except Exception as e:
            st.error(f"Simulation failed: {e}")
            return

        if not sim1 or not sim2:
            st.error("Simulation returned no results. Check golfer data and try again.")
            return

        # Results
        st.subheader("Comparison Results")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### {golfer1}")
            st.metric("Expected Value", f"${sim1.mean_earnings:,.0f}")
            st.metric("Win %", f"{sim1.win_rate*100:.2f}%")
            st.metric("Top-10 %", f"{sim1.top_10_rate*100:.1f}%")
            st.metric("Cut %", f"{sim1.cut_rate*100:.0f}%")
            st.metric("Upside (90th)", f"${sim1.percentile_90:,.0f}")

        with col2:
            st.markdown(f"### {golfer2}")
            ev_diff = sim2.mean_earnings - sim1.mean_earnings
            st.metric("Expected Value", f"${sim2.mean_earnings:,.0f}",
                     delta=f"${ev_diff:+,.0f}")
            st.metric("Win %", f"{sim2.win_rate*100:.2f}%")
            st.metric("Top-10 %", f"{sim2.top_10_rate*100:.1f}%")
            st.metric("Cut %", f"{sim2.cut_rate*100:.0f}%")
            st.metric("Upside (90th)", f"${sim2.percentile_90:,.0f}")

        # Recommendation
        if sim1.mean_earnings > sim2.mean_earnings:
            winner = golfer1
            diff = sim1.mean_earnings - sim2.mean_earnings
        else:
            winner = golfer2
            diff = sim2.mean_earnings - sim1.mean_earnings

        st.success(f"**Recommendation: {winner}** (+${diff:,.0f} EV)")


def show_planner():
    """Season planning tool with full grid and assignment capability (Phase 2.3)."""
    st.title("Season Planner")

    db = st.session_state.db
    strategy = st.session_state.strategy

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Season Grid", "Auto-Optimize", "Projections"])

    with tab1:
        show_season_grid(db, strategy)

    with tab2:
        show_auto_optimize(db, strategy)

    with tab3:
        show_season_projections(db, strategy)


def show_season_grid(db, strategy):
    """Display the full season schedule grid with assignment capability."""
    st.subheader("Season Assignment Grid")

    schedule = get_schedule()
    today = date.today()

    # Get used golfers
    used_golfers = set(db.get_used_golfers())

    # Get season plan from database
    plan_entries = db.get_season_plan()
    plan_by_tournament = {p.tournament_name: p for p in plan_entries}

    # Get all golfers for dropdown
    all_golfers = db.get_all_golfers()
    golfer_names = ["-- Not Assigned --"] + [g.name for g in all_golfers if g.name not in used_golfers]

    # Check for conflicts
    conflicts = db.get_season_plan_conflicts()
    if conflicts:
        st.error(f"**CONFLICTS DETECTED:** The following golfers are assigned to multiple tournaments: {', '.join(conflicts)}")

    # Build the grid data
    grid_data = []
    for tournament in schedule:
        is_past = tournament.date < today
        plan_entry = plan_by_tournament.get(tournament.name)

        # Determine assignment status
        if is_past:
            # Check if we have actual pick data
            picks = db.get_all_picks()
            pick = next((p for p in picks if p.tournament_name == tournament.name), None)
            assigned = pick.golfer_name if pick else "-"
            status = "COMPLETED" if pick else "MISSED"
        else:
            assigned = plan_entry.golfer_name if plan_entry and plan_entry.golfer_name else ""
            status = "TENTATIVE" if assigned else "OPEN"

        # Check if assigned golfer is in conflicts
        is_conflict = assigned in conflicts if assigned else False

        event_type = []
        if tournament.is_major:
            event_type.append("MAJOR")
        if tournament.is_signature:
            event_type.append("SIG")
        if tournament.is_opposite_field:
            event_type.append("OPP")

        grid_data.append({
            "Date": tournament.date.strftime("%m/%d"),
            "Tournament": tournament.name[:35],
            "Type": "/".join(event_type) if event_type else "-",
            "Purse": f"${tournament.purse/1_000_000:.1f}M",
            "Assigned": assigned,
            "Status": status,
            "EV": f"${plan_entry.projected_ev:,.0f}" if plan_entry and plan_entry.projected_ev else "-",
            "Conflict": "YES" if is_conflict else "-",
            "is_past": is_past,
            "tournament_obj": tournament,
        })

    # Display as editable grid for future events
    st.markdown("### Upcoming Tournaments")
    upcoming_data = [d for d in grid_data if not d["is_past"]]

    if not upcoming_data:
        st.info("No upcoming tournaments")
        return

    # Create columns for assignment
    for i, row in enumerate(upcoming_data[:20]):  # Show first 20 upcoming
        tournament = row["tournament_obj"]
        with st.expander(f"{row['Date']} - {row['Tournament']} ({row['Type']}) - {row['Purse']}", expanded=False):
            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                # Golfer selection dropdown
                current_idx = 0
                if row["Assigned"] and row["Assigned"] != "":
                    try:
                        current_idx = golfer_names.index(row["Assigned"])
                    except ValueError:
                        golfer_names.append(row["Assigned"])
                        current_idx = len(golfer_names) - 1

                selected = st.selectbox(
                    "Assign Golfer",
                    golfer_names,
                    index=current_idx,
                    key=f"assign_{tournament.name}"
                )

            with col2:
                if st.button("Save", key=f"save_{tournament.name}"):
                    golfer_name = selected if selected != "-- Not Assigned --" else None

                    # Calculate projected EV if golfer assigned
                    projected_ev = 0.0
                    if golfer_name:
                        golfer = db.get_golfer(golfer_name)
                        if golfer:
                            try:
                                recs = strategy.get_recommendations(tournament, top_n=50, available_only=False)
                                rec = next((r for r in recs if r.golfer.name == golfer_name), None)
                                if rec:
                                    projected_ev = rec.expected_value
                            except Exception:
                                pass

                    db.save_season_plan_entry(
                        tournament_name=tournament.name,
                        tournament_date=tournament.date,
                        golfer_name=golfer_name,
                        projected_ev=projected_ev,
                    )
                    st.success(f"Saved: {golfer_name or 'Cleared'}")
                    st.rerun()

            with col3:
                if row["Conflict"] == "YES":
                    st.error("CONFLICT")
                elif row["Assigned"]:
                    st.success("Assigned")

            # Show current assignment info
            if row["Assigned"] and row["Assigned"] != "":
                st.caption(f"Current: {row['Assigned']} | Projected EV: {row['EV']}")

    # Summary stats
    st.divider()
    st.subheader("Plan Summary")

    assigned_count = sum(1 for d in upcoming_data if d["Assigned"] and d["Assigned"] != "")
    total_upcoming = len(upcoming_data)
    projected_earnings = db.get_projected_season_earnings()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tournaments Assigned", f"{assigned_count}/{total_upcoming}")
    with col2:
        st.metric("Projected Earnings", f"${projected_earnings:,.0f}")
    with col3:
        st.metric("Conflicts", len(conflicts))
    with col4:
        st.metric("Golfers Used", len(used_golfers))


def show_auto_optimize(db, strategy):
    """Auto-optimize season plan with AI suggestions."""
    st.subheader("Auto-Optimize Season Plan")

    st.markdown("""
    Generate optimized golfer assignments for remaining tournaments.
    The optimizer will:
    - Reserve elites for majors and high-value events
    - Assign mid-tier golfers to regular events
    - Avoid conflicts and already-used golfers
    """)

    # Risk level slider
    risk = st.slider("Risk Level", 1, 10, 5,
                    help="1-3: Conservative, 4-6: Balanced, 7-10: Aggressive")

    col1, col2 = st.columns(2)
    with col1:
        reserve_elites = st.number_input("Reserve elites for majors", min_value=0, max_value=10, value=4)

    with col2:
        optimize_scope = st.selectbox("Optimize", ["All Remaining", "Next 10 Events", "Majors Only"])

    if st.button("Generate Optimized Plan", type="primary"):
        with st.spinner("Running optimization..."):
            plan = strategy.get_season_plan(risk_level=risk, remaining_elites=reserve_elites)

        st.success("Optimization complete!")

        # Display elite reservations
        st.subheader("Elite Reservations (Majors)")
        if plan.get('elite_reservation'):
            for r in plan['elite_reservation']:
                st.info(f"**{r['tournament']}**: {r['golfer']} - {r['reasoning']}")
        else:
            st.caption("No elite reservations available")

        # Display signature picks
        st.subheader("Signature Event Picks")
        if plan.get('recommendations', {}).get('signatures'):
            data = plan['recommendations']['signatures']
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.caption("No signature event picks")

        # Display regular picks
        st.subheader("Regular Event Picks")
        if plan.get('recommendations', {}).get('regular'):
            data = plan['recommendations']['regular'][:15]
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.caption("No regular event picks")

        # Apply to season plan button
        if st.button("Apply Suggestions to Season Plan"):
            applied = 0

            # Apply elite reservations
            for r in plan.get('elite_reservation', []):
                tournament = get_tournament_by_name(r['tournament'])
                if tournament:
                    db.save_season_plan_entry(
                        tournament_name=r['tournament'],
                        tournament_date=tournament.date,
                        golfer_name=r['golfer'],
                        is_tentative=True,
                        notes="Auto-optimized elite reservation",
                    )
                    applied += 1

            # Apply signature picks
            for r in plan.get('recommendations', {}).get('signatures', []):
                tournament = get_tournament_by_name(r['tournament'])
                if tournament:
                    db.save_season_plan_entry(
                        tournament_name=r['tournament'],
                        tournament_date=tournament.date,
                        golfer_name=r['golfer'],
                        is_tentative=True,
                        notes="Auto-optimized signature pick",
                    )
                    applied += 1

            # Apply regular picks
            for r in plan.get('recommendations', {}).get('regular', []):
                tournament = get_tournament_by_name(r['tournament'])
                if tournament:
                    db.save_season_plan_entry(
                        tournament_name=r['tournament'],
                        tournament_date=tournament.date,
                        golfer_name=r['golfer'],
                        is_tentative=True,
                        notes="Auto-optimized regular pick",
                    )
                    applied += 1

            st.success(f"Applied {applied} suggestions to season plan!")
            st.rerun()


def show_season_projections(db, strategy):
    """Show season earnings projections and analysis."""
    st.subheader("Season Projections")

    # Get season plan
    plan_entries = db.get_season_plan()
    schedule = get_schedule()
    today = date.today()

    # Filter to assigned future events
    future_assignments = [p for p in plan_entries if p.golfer_name and p.tournament_date >= today]

    if not future_assignments:
        st.info("No future assignments in season plan. Go to Season Grid to assign golfers.")
        return

    # Calculate projections
    total_projected = sum(p.projected_ev for p in future_assignments)
    actual_earnings = db.get_total_earnings()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Actual Earnings (YTD)", f"${actual_earnings:,}")
    with col2:
        st.metric("Projected Future", f"${total_projected:,.0f}")
    with col3:
        st.metric("Projected Total", f"${actual_earnings + total_projected:,.0f}")

    st.divider()

    # Show projection breakdown
    st.subheader("Projection Breakdown")

    data = []
    for entry in future_assignments:
        tournament = get_tournament_by_name(entry.tournament_name)
        data.append({
            "Date": entry.tournament_date.strftime("%m/%d"),
            "Tournament": entry.tournament_name[:30],
            "Golfer": entry.golfer_name,
            "Projected EV": entry.projected_ev,
            "Type": "MAJOR" if tournament and tournament.is_major else ("SIG" if tournament and tournament.is_signature else "-"),
            "Status": "Tentative" if entry.is_tentative else "Locked",
        })

    if data:
        df = pd.DataFrame(data)
        st.dataframe(
            df.style.format({"Projected EV": "${:,.0f}"}),
            use_container_width=True
        )

        # Chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[d["Tournament"][:15] for d in data],
            y=[d["Projected EV"] for d in data],
            marker_color=['#FFD700' if d["Type"] == "MAJOR" else '#4CAF50' for d in data]
        ))
        fig.update_layout(
            title="Projected Earnings by Tournament",
            xaxis_title="Tournament",
            yaxis_title="Projected EV ($)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Confidence intervals (if we have enough data)
    st.subheader("Projection Confidence")
    st.markdown("""
    **Note:** Projections are based on Monte Carlo simulations. Actual results will vary.
    - **10th percentile (downside):** ~60% of projected
    - **Median:** ~80% of projected
    - **90th percentile (upside):** ~150% of projected
    """)

    low = total_projected * 0.6
    median = total_projected * 0.8
    high = total_projected * 1.5

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Downside (10th)", f"${actual_earnings + low:,.0f}")
    with col2:
        st.metric("Median", f"${actual_earnings + median:,.0f}")
    with col3:
        st.metric("Upside (90th)", f"${actual_earnings + high:,.0f}")


def show_season_simulation():
    """Full season Monte Carlo simulation (Phase 3.1)."""
    st.title("Season Monte Carlo Simulation")

    st.markdown("""
    Run comprehensive Monte Carlo simulations to project your season earnings
    with different allocation strategies. This tool helps you understand the
    range of possible outcomes and compare strategies.
    """)

    db = st.session_state.db
    simulator = st.session_state.simulator
    strategy = st.session_state.strategy

    tab1, tab2 = st.tabs(["Simulate Season Plan", "Compare Strategies"])

    with tab1:
        show_simulate_season_plan(db, simulator)

    with tab2:
        show_compare_strategies(db, simulator, strategy)


def show_simulate_season_plan(db, simulator):
    """Simulate the current season plan."""
    st.subheader("Simulate Current Season Plan")

    # Get season plan
    plan_entries = db.get_season_plan()
    schedule = get_schedule()
    today = date.today()

    # Filter to assigned future events
    future_assignments = [p for p in plan_entries if p.golfer_name and p.tournament_date >= today]

    if not future_assignments:
        st.warning("No future assignments in season plan. Go to Season Planner to assign golfers first.")
        return

    st.info(f"Found {len(future_assignments)} planned picks for simulation")

    # Build planned picks list
    planned_picks = []
    for entry in future_assignments:
        golfer = db.get_golfer(entry.golfer_name)
        tournament = get_tournament_by_name(entry.tournament_name)
        if golfer and tournament:
            planned_picks.append((golfer, tournament))

    # Simulation settings
    n_sims = st.slider("Number of Simulations", 1000, 50000, 10000, step=1000,
                      help="More simulations = more accurate but slower")

    if st.button("Run Season Simulation", type="primary"):
        with st.spinner(f"Running {n_sims:,} simulations across {len(planned_picks)} tournaments..."):
            results = simulator.simulate_full_season(planned_picks, n_sims)

        st.success("Simulation complete!")

        # Summary metrics
        st.subheader("Season Projection Summary")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Earnings", f"${results['mean_earnings']:,.0f}")
        with col2:
            st.metric("Median Earnings", f"${results['median_earnings']:,.0f}")
        with col3:
            st.metric("Expected Wins", f"{results['mean_wins']:.1f}")
        with col4:
            st.metric("Expected Top-10s", f"{results['mean_top_10s']:.1f}")

        # Confidence intervals
        st.subheader("Earnings Confidence Intervals")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("5th Percentile (Worst Case)", f"${results['percentile_5']:,.0f}")
            st.metric("10th Percentile", f"${results['percentile_10']:,.0f}")
        with col2:
            st.metric("25th Percentile", f"${results['percentile_25']:,.0f}")
            st.metric("50th Percentile (Median)", f"${results['percentile_50']:,.0f}")
        with col3:
            st.metric("75th Percentile", f"${results['percentile_75']:,.0f}")
            st.metric("95th Percentile (Best Case)", f"${results['percentile_95']:,.0f}")

        # Win distribution
        st.subheader("Win Probability Distribution")

        win_dist = results['win_distribution']
        fig_wins = go.Figure()
        fig_wins.add_trace(go.Bar(
            x=list(win_dist.keys()),
            y=list(win_dist.values()),
            text=[f"{v/n_sims*100:.1f}%" for v in win_dist.values()],
            textposition='outside',
            marker_color=['#4CAF50' if k == '5_plus_wins' else '#2196F3' for k in win_dist.keys()]
        ))
        fig_wins.update_layout(
            title="Probability of Winning X Tournaments",
            xaxis_title="Number of Wins",
            yaxis_title="Number of Simulations",
            height=400
        )
        st.plotly_chart(fig_wins, use_container_width=True)

        # Top-10 distribution
        st.subheader("Top-10 Distribution")

        top10_dist = results['top_10_distribution']
        fig_top10 = go.Figure()
        fig_top10.add_trace(go.Pie(
            labels=list(top10_dist.keys()),
            values=list(top10_dist.values()),
            hole=0.3
        ))
        fig_top10.update_layout(title="Top-10 Finishes Distribution", height=400)
        st.plotly_chart(fig_top10, use_container_width=True)

        # Per-tournament breakdown
        st.subheader("Per-Tournament Projections")

        tournament_data = []
        for t_name, t_results in results['tournament_results'].items():
            tournament_data.append({
                "Tournament": t_name[:30],
                "Golfer": t_results['golfer'],
                "Mean EV": t_results['mean_earnings'],
                "Win Prob": t_results['win_prob'] * 100,
                "Top-10 Prob": t_results['top_10_prob'] * 100,
                "Downside": t_results['percentile_10'],
                "Upside": t_results['percentile_90'],
            })

        if tournament_data:
            df = pd.DataFrame(tournament_data)
            st.dataframe(
                df.style.format({
                    "Mean EV": "${:,.0f}",
                    "Win Prob": "{:.1f}%",
                    "Top-10 Prob": "{:.1f}%",
                    "Downside": "${:,.0f}",
                    "Upside": "${:,.0f}",
                }),
                use_container_width=True
            )


def show_compare_strategies(db, simulator, strategy):
    """Compare different allocation strategies."""
    st.subheader("Compare Allocation Strategies")

    st.markdown("""
    Compare different pick allocation strategies through simulation:
    - **EV Maximization**: Pick highest expected value golfer for each event
    - **Conservative**: Favor consistent performers (lower variance)
    - **Aggressive**: Favor high-upside picks (maximize 90th percentile)
    """)

    # Get available golfers and remaining tournaments
    used_golfers = set(db.get_used_golfers())
    all_golfers = [g for g in db.get_all_golfers() if g.name not in used_golfers]

    schedule = get_schedule()
    today = date.today()
    remaining_tournaments = [t for t in schedule if t.date >= today]

    if not all_golfers:
        st.warning("No available golfers. Please sync data first.")
        return

    if not remaining_tournaments:
        st.warning("No remaining tournaments.")
        return

    st.info(f"Comparing strategies across {len(remaining_tournaments)} tournaments with {len(all_golfers)} available golfers")

    # Simulation settings
    n_sims = st.slider("Simulations per Strategy", 1000, 20000, 5000, step=1000,
                      key="compare_sims",
                      help="More simulations = more accurate but slower")

    strategies_to_compare = st.multiselect(
        "Strategies to Compare",
        ["ev_max", "conservative", "aggressive"],
        default=["ev_max", "conservative", "aggressive"]
    )

    if st.button("Compare Strategies", type="primary"):
        with st.spinner(f"Running strategy comparison ({len(strategies_to_compare)} strategies x {n_sims:,} simulations each)..."):
            results = simulator.compare_allocation_strategies(
                all_golfers,
                remaining_tournaments[:20],  # Limit to first 20 for speed
                strategies_to_compare,
                n_sims
            )

        st.success("Comparison complete!")

        # Summary comparison
        st.subheader("Strategy Comparison Summary")

        comparison_data = []
        for strat_name, strat_results in results.items():
            comparison_data.append({
                "Strategy": strat_name.replace("_", " ").title(),
                "Mean Earnings": strat_results['mean_earnings'],
                "Median Earnings": strat_results['median_earnings'],
                "Std Dev": strat_results['std_earnings'],
                "10th Percentile": strat_results['percentile_10'],
                "90th Percentile": strat_results['percentile_90'],
                "Expected Wins": strat_results['mean_wins'],
                "Expected Top-10s": strat_results['mean_top_10s'],
            })

        df = pd.DataFrame(comparison_data)
        st.dataframe(
            df.style.format({
                "Mean Earnings": "${:,.0f}",
                "Median Earnings": "${:,.0f}",
                "Std Dev": "${:,.0f}",
                "10th Percentile": "${:,.0f}",
                "90th Percentile": "${:,.0f}",
                "Expected Wins": "{:.2f}",
                "Expected Top-10s": "{:.1f}",
            }),
            use_container_width=True
        )

        # Bar chart comparison
        fig = go.Figure()

        for i, strat in enumerate(comparison_data):
            fig.add_trace(go.Bar(
                name=strat["Strategy"],
                x=["Mean", "Median", "10th Pct", "90th Pct"],
                y=[strat["Mean Earnings"], strat["Median Earnings"],
                   strat["10th Percentile"], strat["90th Percentile"]],
            ))

        fig.update_layout(
            title="Strategy Earnings Comparison",
            barmode='group',
            yaxis_title="Earnings ($)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

        # Recommendation
        best_mean = max(comparison_data, key=lambda x: x["Mean Earnings"])
        best_median = max(comparison_data, key=lambda x: x["Median Earnings"])
        best_upside = max(comparison_data, key=lambda x: x["90th Percentile"])
        safest = max(comparison_data, key=lambda x: x["10th Percentile"])

        st.subheader("Recommendations")
        st.success(f"**Highest Expected Value:** {best_mean['Strategy']} (${best_mean['Mean Earnings']:,.0f})")
        if best_median != best_mean:
            st.info(f"**Most Likely Outcome:** {best_median['Strategy']} (${best_median['Median Earnings']:,.0f})")
        st.info(f"**Highest Upside:** {best_upside['Strategy']} (90th pct: ${best_upside['90th Percentile']:,.0f})")
        st.info(f"**Safest Floor:** {safest['Strategy']} (10th pct: ${safest['10th Percentile']:,.0f})")


def show_multi_entry():
    """Multi-entry strategy support and segment optimization (Phase 3.2 & 3.3)."""
    st.title("Multi-Entry Strategy")

    db = st.session_state.db
    strategy = st.session_state.strategy

    tab1, tab2, tab3 = st.tabs(["Multi-Entry Picks", "Hedge Calculator", "Segment Optimization"])

    with tab1:
        show_multi_entry_picks(db, strategy)

    with tab2:
        show_hedge_calculator(db, strategy)

    with tab3:
        show_segment_optimization(db, strategy)


def show_multi_entry_picks(db, strategy):
    """Generate picks for multiple entries."""
    st.subheader("Multi-Entry Pick Recommendations")

    st.markdown("""
    If you have multiple entries in your One and Done league, you should
    **diversify your picks** to maximize your chances of winning at least one entry.

    This tool recommends different picks for each entry to create optimal hedging.
    """)

    # Tournament selector
    schedule = get_schedule()
    upcoming = [t for t in schedule if t.date >= date.today()]
    tournament_names = [t.name for t in upcoming]

    selected = st.selectbox("Select Tournament", tournament_names, key="multi_entry_tournament")
    tournament = get_tournament_by_name(selected) if selected else get_next_tournament()

    if not tournament:
        st.warning("No tournament selected")
        return

    # Number of entries
    num_entries = st.slider("Number of Entries", 2, 5, 2)

    if st.button("Generate Multi-Entry Picks", type="primary"):
        with st.spinner(f"Generating diversified picks for {num_entries} entries..."):
            multi_recs = strategy.get_multi_entry_recommendations(
                tournament, num_entries, top_n_per_entry=3
            )

        if not multi_recs:
            st.error("Could not generate recommendations. Please sync data first.")
            return

        st.success(f"Generated picks for {num_entries} entries!")

        # Display entry recommendations
        for entry_num, recs in multi_recs.items():
            st.subheader(f"Entry {entry_num}")

            if not recs:
                st.warning("No picks available for this entry")
                continue

            for i, rec in enumerate(recs, 1):
                col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

                with col1:
                    rank_emoji = ["", "", ""][min(i-1, 2)]
                    st.markdown(f"**{rank_emoji} {rec.golfer.name}**")

                with col2:
                    st.metric("EV", f"${rec.expected_value:,.0f}")

                with col3:
                    st.metric("Win %", f"{rec.golfer.win_probability*100:.1f}%")

                with col4:
                    st.metric("OWGR", rec.golfer.owgr)

            st.divider()

        # Hedging analysis
        st.subheader("Hedging Analysis")

        # Calculate combined coverage
        all_golfers = set()
        for recs in multi_recs.values():
            for rec in recs:
                all_golfers.add(rec.golfer.name)

        st.info(f"**Diversification:** {len(all_golfers)} unique golfers across {num_entries} entries")

        # Win probability coverage
        total_win_prob = 0
        for recs in multi_recs.values():
            if recs:
                total_win_prob += recs[0].golfer.win_probability

        st.metric("Combined Win Probability (top picks)",
                 f"{min(total_win_prob, 1.0)*100:.1f}%",
                 help="Sum of win probabilities for top pick in each entry (capped at 100%)")


def show_hedge_calculator(db, strategy):
    """Calculate hedge picks for a primary pick."""
    st.subheader("Hedge Calculator")

    st.markdown("""
    Given your primary pick, this tool finds the best hedging options
    that complement your selection by providing diversification.
    """)

    # Tournament selector
    schedule = get_schedule()
    upcoming = [t for t in schedule if t.date >= date.today()]
    tournament_names = [t.name for t in upcoming]

    selected = st.selectbox("Select Tournament", tournament_names, key="hedge_tournament")
    tournament = get_tournament_by_name(selected) if selected else None

    if not tournament:
        st.warning("No tournament selected")
        return

    # Primary pick selector
    all_golfers = db.get_all_golfers()
    golfer_names = [g.name for g in all_golfers]

    primary_pick = st.selectbox("Your Primary Pick", golfer_names, key="primary_pick")

    num_hedges = st.slider("Number of Hedge Picks", 1, 5, 2)

    if st.button("Find Hedge Picks", type="primary"):
        with st.spinner("Finding optimal hedges..."):
            hedge_picks = strategy.get_hedging_picks(tournament, primary_pick, num_hedges)

        if not hedge_picks:
            st.error("Could not find hedge picks. Please sync data first.")
            return

        st.success(f"Found {len(hedge_picks)} hedge picks!")

        # Display primary pick
        st.subheader("Your Primary Pick")
        primary_golfer = db.get_golfer(primary_pick)
        if primary_golfer:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**{primary_golfer.name}**")
            with col2:
                st.metric("OWGR", primary_golfer.owgr)
            with col3:
                st.metric("Win %", f"{primary_golfer.win_probability*100:.1f}%" if primary_golfer.win_probability else "N/A")

        # Display hedge picks
        st.subheader("Recommended Hedges")

        for i, rec in enumerate(hedge_picks, 1):
            tier = strategy.classify_golfer_tier(rec.golfer)

            with st.expander(f"Hedge {i}: {rec.golfer.name}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Expected Value", f"${rec.expected_value:,.0f}")

                with col2:
                    st.metric("OWGR", rec.golfer.owgr)

                with col3:
                    st.metric("Tier", tier.upper())

                with col4:
                    if rec.hedge_bonus > 0:
                        st.metric("Hedge Bonus", f"+{rec.hedge_bonus*100:.0f}%")
                    else:
                        st.metric("Hedge Bonus", "-")

                st.caption(f"**Reasoning:** {rec.reasoning[:200]}...")


def show_segment_optimization(db, strategy):
    """Segment optimization for the season."""
    st.subheader("Season Segment Optimization")

    st.markdown("""
    **Strategy:** Divide the season into segments and optimize elite deployment
    across each segment. This ensures you don't use all your elites too early
    or miss opportunities at key events.
    """)

    # Number of segments
    num_segments = st.slider("Number of Segments", 4, 8, 6,
                            help="Season divided into this many segments")

    if st.button("Generate Segment Plan", type="primary"):
        with st.spinner("Optimizing segments..."):
            segment_plan = strategy.get_segment_optimization(num_segments)

        if "error" in segment_plan:
            st.error(segment_plan["error"])
            return

        st.success("Segment optimization complete!")

        # Summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Segments", segment_plan["num_segments"])
        with col2:
            st.metric("Remaining Events", segment_plan["total_remaining_events"])
        with col3:
            st.metric("Available Elites", segment_plan["available_elites"])
        with col4:
            st.metric("Available Mid-Tier", segment_plan["available_mid_tier"])

        # Strategy notes
        st.subheader("Strategy Notes")
        for note in segment_plan["strategy_notes"]:
            st.info(note)

        # Segment breakdown
        st.subheader("Segment Breakdown")

        for segment in segment_plan["segments"]:
            segment_type = []
            if segment["has_major"]:
                segment_type.append("MAJOR")
            if segment["has_signature"]:
                segment_type.append("SIGNATURE")
            type_str = " | ".join(segment_type) if segment_type else "Regular"

            with st.expander(
                f"Segment {segment['segment_num']}: {segment['start_date']} - {segment['end_date']} ({type_str})",
                expanded=segment["has_major"]
            ):
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Events", segment["num_events"])
                with col2:
                    st.metric("Total Purse", f"${segment['total_purse']/1_000_000:.1f}M")
                with col3:
                    if segment["elite_pick"]:
                        st.metric("Elite Pick", segment["elite_pick"])
                    else:
                        st.metric("Elite Pick", "None recommended")

                if segment["elite_tournament"]:
                    st.success(f"**Deploy Elite at:** {segment['elite_tournament']}")

                # Tournament list
                st.markdown("**Tournaments in this segment:**")
                for t_name in segment["tournament_names"]:
                    tournament = get_tournament_by_name(t_name)
                    if tournament:
                        emoji = "" if tournament.is_major else ("" if tournament.is_signature else "")
                        st.markdown(f"- {emoji} {t_name}")

        # Elite allocation summary
        st.subheader("Elite Allocation Summary")

        if segment_plan["elite_allocations"]:
            for tournament, golfer in segment_plan["elite_allocations"].items():
                st.success(f"**{tournament}:** {golfer}")
        else:
            st.info("No elite allocations recommended yet. Sync data to get recommendations.")


def show_standings():
    """League standings view."""
    st.title("League Standings")

    db = st.session_state.db
    standings = db.get_latest_standings()

    if not standings:
        st.warning("No standings data. Please update from the site.")
        return

    config = get_config()
    my_username = config.site_username.lower()

    # Find user's position
    my_standing = None
    for s in standings:
        if s.username.lower() == my_username:
            my_standing = s
            break

    if my_standing:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Your Rank", f"#{my_standing.rank}")
        with col2:
            st.metric("Your Earnings", f"${my_standing.total_earnings:,}")
        with col3:
            st.metric("Cuts Made", my_standing.cuts_made)

    st.divider()

    # Full standings table
    data = []
    for s in standings:
        is_me = s.username.lower() == my_username
        data.append({
            "Rank": s.rank,
            "Player": s.player_name + (" " if is_me else ""),
            "Earnings": s.total_earnings,
            "Cuts": s.cuts_made
        })

    df = pd.DataFrame(data)
    st.dataframe(
        df.style.format({"Earnings": "${:,}"}),
        use_container_width=True,
        height=500
    )


def show_settings():
    """Settings page."""
    st.title("Settings")

    config = get_config()

    st.subheader("Data Golf API")
    api_key = st.text_input("API Key", value=config.datagolf_api_key, type="password")

    st.subheader("League Settings")
    st.text_input("Username", value=config.site_username)
    st.text_input("League Name", value=config.league_name)

    st.subheader("Strategy Settings")
    risk = st.slider("Default Risk Level", 1, 10, config.risk_level)

    if st.button("Save Settings"):
        config.datagolf_api_key = api_key
        config.risk_level = risk
        config.save_to_env()
        st.success("Settings saved!")

    st.divider()

    st.subheader("Data Management")

    # Data status metrics
    db = st.session_state.db
    api = st.session_state.api

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Golfers in DB", db.get_golfer_count())
    with col2:
        st.metric("Valid OWGR", db.get_valid_owgr_count())
    with col3:
        st.metric("Picks Made", db.get_picks_count())
    with col4:
        st.metric("Total Earnings", f"${db.get_total_earnings():,}")

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("Update from API", type="primary"):
            with st.spinner("Fetching data from Data Golf API..."):
                try:
                    count = api.sync_golfers_to_db()
                    preds = api.get_pre_tournament_predictions()
                    st.session_state.data_sync_needed = False
                    st.success(f"Synced {count} golfers and {len(preds)} predictions!")
                except Exception as e:
                    st.error(f"Sync failed: {e}")
                    if api.last_error:
                        st.caption(f"Details: {api.last_error}")

    with col2:
        if st.button("Clear All Cache"):
            cleared = db.clear_all_api_cache()
            db.clear_expired_cache()
            st.success(f"Cleared {cleared} cached API responses!")

    with col3:
        if st.button("Clear Simulation Cache"):
            cleared = db.clear_simulation_cache()
            st.success(f"Cleared {cleared} cached simulations. Recommendations will recalculate.")

    with col4:
        if st.button("Check API Health"):
            with st.spinner("Checking API..."):
                try:
                    healthy = api.health_check()
                    if healthy:
                        st.success("API connection healthy!")
                    else:
                        st.error("API check failed. Verify your API key.")
                except Exception as e:
                    st.error(f"API check failed: {e}")

    # Show last error if any
    if api.last_error:
        st.warning(f"Last API Error: {api.last_error}")

    st.divider()

    # Import from Fantasy Site
    st.subheader("Import from BuzzFantasyGolf.com")
    st.markdown("""
    Import your standings, picks history, and available golfers from the fantasy site.
    This requires your login credentials to be configured above.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Import Standings & Picks", type="primary"):
            with st.spinner("Connecting to buzzfantasygolf.com..."):
                try:
                    from scraper import Scraper
                    scraper = Scraper(headless=True)

                    # Login
                    if not scraper.login():
                        st.error("Login failed. Check your credentials in Settings.")
                    else:
                        st.info("Logged in successfully!")

                        # Get standings
                        with st.spinner("Fetching standings..."):
                            standings = scraper.get_standings()
                            if standings:
                                st.success(f"Imported {len(standings)} standings entries!")
                            else:
                                st.warning("No standings found")

                        # Get my picks
                        with st.spinner("Fetching your picks history..."):
                            picks = scraper.get_my_picks()
                            if picks:
                                for pick in picks:
                                    db.save_pick(pick)
                                st.success(f"Imported {len(picks)} picks!")
                            else:
                                st.warning("No picks found")

                        # Get available golfers
                        with st.spinner("Fetching available golfers..."):
                            available = scraper.get_available_golfers()
                            if available:
                                db.save_available_golfers(available)
                                st.success(f"Imported {len(available)} available golfers!")
                            else:
                                st.warning("No available golfers found")

                        scraper.close()
                        st.rerun()

                except ImportError as e:
                    st.error(f"Scraper module not available: {e}")
                except Exception as e:
                    st.error(f"Import failed: {e}")

    with col2:
        if st.button("Import Opponent Picks"):
            with st.spinner("Fetching opponent picks..."):
                try:
                    from scraper import Scraper
                    scraper = Scraper(headless=True)

                    if not scraper.login():
                        st.error("Login failed. Check your credentials.")
                    else:
                        opponent_picks = scraper.get_opponent_picks()
                        if opponent_picks:
                            for pick in opponent_picks:
                                db.save_opponent_pick(pick)
                            st.success(f"Imported {len(opponent_picks)} opponent picks!")
                        else:
                            st.warning("No opponent picks found")
                        scraper.close()

                except Exception as e:
                    st.error(f"Import failed: {e}")

    with col3:
        if st.button("Full Sync from Site"):
            with st.spinner("Running full sync..."):
                try:
                    from scraper import Scraper
                    scraper = Scraper(headless=True)
                    results = scraper.sync_all()

                    if "error" in results:
                        st.error(results["error"])
                    else:
                        st.success(f"""
                        **Sync Complete!**
                        - Standings: {results.get('standings', 0)} entries
                        - Available Golfers: {results.get('available_golfers', 0)}
                        - My Picks: {results.get('my_picks', 0)}
                        - Opponent Picks: {results.get('opponent_picks', 0)}
                        """)
                        scraper.close()
                        st.rerun()

                except Exception as e:
                    st.error(f"Full sync failed: {e}")

    # Current standings display
    standings = db.get_latest_standings()
    if standings:
        st.subheader("Current League Standings (from import)")
        my_standing = db.get_my_standing(config.site_username)
        if my_standing:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Your Rank", f"#{my_standing.rank}")
            with col2:
                st.metric("Your Earnings", f"${my_standing.total_earnings:,}")
            with col3:
                st.metric("Cuts Made", my_standing.cuts_made)


def show_learning_insights():
    """Learning Insights page - model confidence, outcome tracking, and learning status."""
    st.title("Learning Insights")
    st.caption("Track model accuracy, record outcomes, and view learning progress")

    learner = st.session_state.learner
    db = st.session_state.db

    # Get dashboard data
    dashboard_data = learner.get_dashboard_data()

    # Tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Outcome Recording", "Model Confidence",
        "Elite Tier Changes", "Course Fit Learning"
    ])

    with tab1:
        st.subheader("Learning System Overview")

        # Key insights
        insights = dashboard_data.get("insights", [])
        if insights:
            st.markdown("### Key Insights")
            for insight in insights:
                impact_color = {
                    "high": "red",
                    "medium": "orange",
                    "low": "blue"
                }.get(insight["impact"], "gray")

                with st.expander(f"{insight['title']} ({insight['category']})", expanded=insight["impact"] == "high"):
                    st.markdown(f"**{insight['description']}**")
                    st.caption(f"Confidence: {insight['confidence']*100:.0f}% | Impact: {insight['impact']}")
        else:
            st.info("No insights yet. Record some tournament outcomes to generate insights.")

        # Summary metrics
        st.markdown("### Performance Summary")
        summary = dashboard_data.get("outcome_summary", {})
        my_picks = summary.get("my_picks_performance", {})

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Predictions", summary.get("total_predictions", 0))
        with col2:
            st.metric("My Picks", my_picks.get("total", 0))
        with col3:
            st.metric("My Wins", my_picks.get("wins", 0))
        with col4:
            st.metric("My Top-10s", my_picks.get("top10s", 0))

        if my_picks.get("total", 0) > 0:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Earnings", f"${my_picks.get('total_earnings', 0):,.0f}")
            with col2:
                st.metric("Predicted EV", f"${my_picks.get('total_predicted_ev', 0):,.0f}")

    with tab2:
        st.subheader("Record Tournament Outcomes")

        # Pending outcomes to record
        pending = dashboard_data.get("pending_outcomes", [])

        if pending:
            st.warning(f"You have {len(pending)} predictions awaiting outcome recording.")

            # Group by tournament
            tournaments = {}
            for p in pending:
                t_name = p.get("tournament_name", "Unknown")
                if t_name not in tournaments:
                    tournaments[t_name] = []
                tournaments[t_name].append(p)

            for t_name, predictions in tournaments.items():
                with st.expander(f"{t_name} ({len(predictions)} predictions)", expanded=True):
                    st.caption(f"Tournament Date: {predictions[0].get('tournament_date', 'Unknown')}")

                    # Option to record all outcomes at once
                    st.markdown("**Quick Record** - Enter results for top picks:")

                    for i, pred in enumerate(predictions[:10]):
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        with col1:
                            st.text(pred.get("golfer_name", "Unknown"))
                        with col2:
                            position = st.number_input(
                                "Pos",
                                min_value=1, max_value=200,
                                value=None,
                                key=f"pos_{t_name}_{i}",
                                label_visibility="collapsed"
                            )
                        with col3:
                            earnings = st.number_input(
                                "Earnings ($)",
                                min_value=0,
                                value=0,
                                key=f"earn_{t_name}_{i}",
                                label_visibility="collapsed"
                            )
                        with col4:
                            made_cut = st.checkbox("Cut", key=f"cut_{t_name}_{i}")

                    if st.button(f"Save Outcomes for {t_name}", key=f"save_{t_name}"):
                        saved = 0
                        for i, pred in enumerate(predictions[:10]):
                            pos_key = f"pos_{t_name}_{i}"
                            earn_key = f"earn_{t_name}_{i}"
                            cut_key = f"cut_{t_name}_{i}"

                            position = st.session_state.get(pos_key)
                            earnings = st.session_state.get(earn_key, 0)
                            made_cut = st.session_state.get(cut_key, False)

                            if position:
                                learner.outcome_tracker.record_outcome(
                                    golfer_name=pred.get("golfer_name"),
                                    tournament_name=t_name,
                                    actual_position=position,
                                    actual_earnings=earnings,
                                    made_cut=made_cut or position <= 65
                                )
                                saved += 1

                        if saved > 0:
                            st.success(f"Saved {saved} outcomes!")
                            st.rerun()
        else:
            st.success("All predictions have outcomes recorded!")

        # Manual outcome entry
        st.divider()
        st.subheader("Manual Outcome Entry")

        with st.form("manual_outcome"):
            col1, col2 = st.columns(2)
            with col1:
                golfer_name = st.text_input("Golfer Name")
                tournament_name = st.text_input("Tournament Name")
            with col2:
                position = st.number_input("Finish Position", min_value=1, max_value=200)
                earnings = st.number_input("Earnings ($)", min_value=0)

            made_cut = st.checkbox("Made Cut")

            if st.form_submit_button("Record Outcome"):
                if golfer_name and tournament_name:
                    learner.outcome_tracker.record_outcome(
                        golfer_name=golfer_name,
                        tournament_name=tournament_name,
                        actual_position=position,
                        actual_earnings=earnings,
                        made_cut=made_cut
                    )
                    st.success(f"Recorded outcome for {golfer_name}")

    with tab3:
        st.subheader("Model Confidence Trends")

        confidence_data = dashboard_data.get("model_confidence", {})

        if confidence_data:
            # Display metrics
            col1, col2, col3 = st.columns(3)

            metrics_display = {
                "win_prob_brier": ("Win Prob Brier", "Lower is better"),
                "top10_prob_brier": ("Top-10 Prob Brier", "Lower is better"),
                "ev_mae": ("EV Mean Abs Error", "Lower is better"),
            }

            for i, (metric, (label, help_text)) in enumerate(metrics_display.items()):
                with [col1, col2, col3][i % 3]:
                    data = confidence_data.get(metric, {})
                    current = data.get("current")
                    trend = data.get("trend", "unknown")

                    trend_icon = {
                        "improving": "improving",
                        "declining": "declining",
                        "stable": "stable",
                    }.get(trend, "")

                    if current is not None:
                        st.metric(
                            label,
                            f"{current:.4f}",
                            delta=trend_icon,
                            delta_color="inverse" if trend == "improving" else "normal"
                        )
                        st.caption(help_text)
                    else:
                        st.metric(label, "N/A")

            # Trend chart
            st.divider()
            st.subheader("Accuracy Over Time")

            # Get history for charting
            accuracy_history = db.get_model_accuracy_history(days=180)
            if accuracy_history:
                df = pd.DataFrame(accuracy_history)
                df['recorded_at'] = pd.to_datetime(df['recorded_at'])

                # Group by metric for separate lines
                fig = px.line(
                    df,
                    x='recorded_at',
                    y='metric_value',
                    color='metric_name',
                    title='Model Accuracy Trends'
                )
                fig.update_layout(
                    xaxis_title="Date",
                    yaxis_title="Metric Value (lower is better)"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Record more outcomes to see accuracy trends over time.")
        else:
            st.info("No confidence metrics available yet. Record tournament outcomes to track model accuracy.")

    with tab4:
        st.subheader("Elite Tier Dynamic Adjustments")

        tier_changes = dashboard_data.get("elite_tier_changes", [])

        if tier_changes:
            st.markdown("""
            The system learns from actual performance to suggest tier adjustments.
            Golfers performing above their tier get promoted, underperformers get flagged.
            """)

            # Promotions
            promotions = [t for t in tier_changes if t.get("learned_tier", 0) < t.get("static_tier", 0)
                         or (t.get("static_tier", 0) == 0 and t.get("learned_tier", 0) > 0)]
            demotions = [t for t in tier_changes if t.get("learned_tier", 0) > t.get("static_tier", 0)
                        and t.get("static_tier", 0) > 0]

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Promotion Candidates")
                if promotions:
                    for p in promotions:
                        tier_names = {0: "Not Elite", 1: "Tier 1 (Save for Majors)", 2: "Tier 2 (Signature Events)"}
                        st.success(f"""
                        **{p['golfer_name']}**
                        - Was: {tier_names.get(p['static_tier'], 'Unknown')}
                        - Should be: {tier_names.get(p['learned_tier'], 'Unknown')}
                        - Wins: {p.get('wins_this_season', 0)} | Top-10s: {p.get('top10s_this_season', 0)}
                        - Earnings: ${p.get('total_earnings', 0):,.0f}
                        """)
                else:
                    st.info("No promotion candidates at this time.")

            with col2:
                st.markdown("### Demotion Candidates")
                if demotions:
                    for d in demotions:
                        tier_names = {0: "Not Elite", 1: "Tier 1", 2: "Tier 2", 3: "Tier 3"}
                        st.warning(f"""
                        **{d['golfer_name']}**
                        - Was: {tier_names.get(d['static_tier'], 'Unknown')}
                        - Should be: {tier_names.get(d['learned_tier'], 'Unknown')}
                        - Wins: {d.get('wins_this_season', 0)} | Top-10s: {d.get('top10s_this_season', 0)}
                        - Earnings: ${d.get('total_earnings', 0):,.0f}
                        """)
                else:
                    st.info("No demotion candidates at this time.")

            # Full tier table
            st.divider()
            st.subheader("All Tracked Elite Tiers")
            all_tiers = db.get_learned_elite_tiers()
            if all_tiers:
                df = pd.DataFrame(all_tiers)
                df = df[['golfer_name', 'static_tier', 'learned_tier', 'performance_score',
                        'wins_this_season', 'top10s_this_season', 'total_earnings', 'tier_confidence']]
                df.columns = ['Golfer', 'Static Tier', 'Learned Tier', 'Perf Score',
                             'Wins', 'Top-10s', 'Earnings', 'Confidence']
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No elite tier adjustments yet. Record tournament outcomes to track player performance.")

    with tab5:
        st.subheader("Course Fit Learning Status")

        course_fit_status = dashboard_data.get("course_fit_status", [])

        if course_fit_status:
            st.markdown("""
            The model learns which skills correlate with success at each course.
            Higher confidence means more weight is given to learned weights vs static weights.
            """)

            # Progress by course
            for course in course_fit_status:
                with st.expander(f"{course['tournament_name']} - Confidence: {course['avg_confidence']*100:.0f}%"):
                    st.progress(course['avg_confidence'])
                    st.caption(f"Sample size: {course['total_sample_size']} data points")

                    # Skill weights
                    if course.get("skills"):
                        skill_data = []
                        for skill in course["skills"]:
                            skill_data.append({
                                "Skill": skill["skill_name"],
                                "Static Weight": f"{skill['static_weight']:.2f}",
                                "Learned Weight": f"{skill['learned_weight']:.2f}",
                                "Confidence": f"{skill['confidence']*100:.0f}%"
                            })
                        st.table(pd.DataFrame(skill_data))
        else:
            st.info("No course fit learning data yet. Record tournament outcomes to calibrate course fit weights.")

        # Opponent patterns
        st.divider()
        st.subheader("Opponent Picking Patterns")

        opponent_patterns = dashboard_data.get("opponent_patterns", [])
        if opponent_patterns:
            df = pd.DataFrame(opponent_patterns)
            df = df[['opponent_name', 'prefers_favorites', 'risk_tolerance',
                    'avg_golfer_ranking', 'total_picks_tracked']]
            df.columns = ['Opponent', 'Prefers Favorites', 'Risk Tolerance',
                         'Avg OWGR Pick', 'Picks Tracked']

            # Format percentages
            df['Prefers Favorites'] = df['Prefers Favorites'].apply(lambda x: f"{x*100:.0f}%")
            df['Risk Tolerance'] = df['Risk Tolerance'].apply(lambda x: f"{x*100:.0f}%")

            st.dataframe(df, use_container_width=True)
        else:
            st.info("No opponent patterns learned yet. Import opponent picks to analyze their patterns.")


if __name__ == "__main__":
    main()
