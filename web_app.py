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
         "What-If Analysis", "Season Planner", "League Standings", "Settings"]
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

    # Number of recommendations
    num_recs = st.slider("Number of recommendations", 5, 20, 10)

    # Generate recommendations
    if st.button("Generate Recommendations", type="primary"):
        strategy = st.session_state.strategy

        with st.spinner("Running 50,000 Monte Carlo simulations per golfer..."):
            recs = strategy.get_recommendations(tournament, top_n=num_recs)

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

                # Show OWGR warning in the expander title if applicable
                owgr_flag = " OWGR RISK" if rec.owgr_warning else ""
                with st.expander(f"{medal} **{rec.golfer.name}** | Win: {win_pct:.1f}% | EV: ${rec.expected_value:,.0f}{owgr_flag}", expanded=(i <= 3)):
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
    """Season planning tool."""
    st.title("Season Planner")

    strategy = st.session_state.strategy

    # Risk level slider
    risk = st.slider("Risk Level", 1, 10, 5,
                    help="1-3: Conservative, 4-6: Balanced, 7-10: Aggressive")

    if st.button("Generate Season Plan", type="primary"):
        with st.spinner("Optimizing season strategy..."):
            plan = strategy.get_season_plan(risk_level=risk)

        st.success("Season plan generated!")

        # Display plan
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Remaining Tournaments", plan['remaining_tournaments'])
            st.metric("Available Elites", plan['available_elites'])

        with col2:
            st.metric("Available Mid-Tier", plan['available_mid_tier'])
            st.metric("Risk Level", f"{risk}/10")

        # Elite reservations
        st.subheader("Elite Reservations (Majors)")
        if plan.get('elite_reservation'):
            for r in plan['elite_reservation']:
                st.info(f"**{r['tournament']}**: {r['golfer']}")
        else:
            st.caption("No elite reservations - sync data to see recommendations")

        # Signature picks
        st.subheader("Signature Event Picks")
        if plan.get('recommendations', {}).get('signatures'):
            data = plan['recommendations']['signatures']
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.caption("No signature event picks available - sync data to see recommendations")

        # Regular picks
        st.subheader("Regular Event Picks")
        if plan.get('recommendations', {}).get('regular'):
            data = plan['recommendations']['regular'][:15]  # Show first 15
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
        else:
            st.caption("No regular event picks available - sync data to see recommendations")


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


if __name__ == "__main__":
    main()
