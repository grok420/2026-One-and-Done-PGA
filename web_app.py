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

from config import get_config, get_schedule, get_next_tournament, get_majors, get_tournament_by_name
from database import Database
from api import DataGolfAPI
from simulator import Simulator
from strategy import Strategy
from models import Tier

# Page config
st.set_page_config(
    page_title="PGA One and Done Optimizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for fun styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
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
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stMetric {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
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


def main():
    """Main application."""
    init_session_state()

    # Header
    st.markdown('<div class="main-header">PGA One and Done Optimizer</div>', unsafe_allow_html=True)

    # Sidebar navigation
    st.sidebar.image("https://www.pgatour.com/content/dam/pgatour/logos/pga-tour-logo.svg", width=150)
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Select Page",
        ["Dashboard", "Pick Recommendations", "Tournament Schedule",
         "What-If Analysis", "Season Planner", "League Standings", "Settings"]
    )

    if page == "Dashboard":
        show_dashboard()
    elif page == "Pick Recommendations":
        show_recommendations()
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

        with st.spinner("Running simulations..."):
            recs = strategy.get_recommendations(next_t, top_n=5)

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

    # Season progress
    st.subheader("Season Progress")

    schedule = get_schedule()
    today = date.today()
    total = len(schedule)
    completed = len([t for t in schedule if t.date < today])

    progress = completed / total if total > 0 else 0
    st.progress(progress, text=f"{completed}/{total} tournaments completed ({progress*100:.0f}%)")


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
    st.info(f"""
    **{tournament.name}** {tier_emoji.get(tournament.tier, '')}

    **Date:** {tournament.date.strftime('%B %d, %Y')} |
    **Purse:** ${tournament.purse:,} |
    **Winner:** ${tournament.winner_share:,} |
    **Tier:** {tournament.tier.name}
    {"| **MAJOR**" if tournament.is_major else ""}{"| **SIGNATURE**" if tournament.is_signature else ""}
    """)

    # Number of recommendations
    num_recs = st.slider("Number of recommendations", 5, 20, 10)

    # Generate recommendations
    if st.button("Generate Recommendations", type="primary"):
        strategy = st.session_state.strategy

        with st.spinner("Running 50,000 Monte Carlo simulations per golfer..."):
            recs = strategy.get_recommendations(tournament, top_n=num_recs)

        if not recs:
            st.error("No recommendations available. Please update data first.")
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

                with st.expander(f"{medal} **{rec.golfer.name}** - EV: ${rec.expected_value:,.0f}", expanded=(i <= 3)):
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Expected Value", f"${rec.expected_value:,.0f}")
                        st.metric("OWGR", rec.golfer.owgr)

                    with col2:
                        win_pct = sim.win_rate * 100 if sim else 0
                        top10_pct = sim.top_10_rate * 100 if sim else 0
                        st.metric("Win %", f"{win_pct:.2f}%")
                        st.metric("Top-10 %", f"{top10_pct:.1f}%")

                    with col3:
                        cut_pct = sim.cut_rate * 100 if sim else 0
                        st.metric("Cut %", f"{cut_pct:.0f}%")
                        st.metric("Confidence", f"{rec.confidence*100:.0f}%")

                    with col4:
                        if sim:
                            st.metric("Upside (90th)", f"${sim.percentile_90:,.0f}")
                            st.metric("Downside (10th)", f"${sim.percentile_10:,.0f}")

                    # Hedge bonus indicator
                    if rec.hedge_bonus > 0:
                        st.success(f"Hedge Bonus: +{rec.hedge_bonus*100:.0f}% (underused by opponents)")

                    # Risk indicator
                    if rec.regret_risk > 0.3:
                        st.warning("Higher regret risk - consider alternatives")
                    elif rec.regret_risk < 0.1:
                        st.success("Low regret risk - solid choice!")

                    st.caption(f"Analysis: {rec.reasoning}")

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
                data.append({
                    "Rank": i,
                    "Golfer": rec.golfer.name,
                    "OWGR": rec.golfer.owgr,
                    "Expected Value": rec.expected_value,
                    "Win %": sim.win_rate * 100 if sim else 0,
                    "Top-10 %": sim.top_10_rate * 100 if sim else 0,
                    "Cut %": sim.cut_rate * 100 if sim else 0,
                    "Upside (90th)": sim.percentile_90 if sim else 0,
                    "Downside (10th)": sim.percentile_10 if sim else 0,
                    "Hedge Bonus": f"+{rec.hedge_bonus*100:.0f}%" if rec.hedge_bonus > 0 else "-",
                    "Confidence": f"{rec.confidence*100:.0f}%"
                })

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

        data.append({
            "Date": t.date.strftime("%b %d, %Y"),
            "Tournament": t.name,
            "Course": t.course,
            "Purse": t.purse,
            "Winner's Share": t.winner_share,
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
    col1, col2, col3, col4 = st.columns(4)

    total_purse = sum(t.purse for t in filtered)
    majors = [t for t in filtered if t.is_major]
    tier1 = [t for t in filtered if t.tier == Tier.TIER_1]

    with col1:
        st.metric("Total Events", len(filtered))
    with col2:
        st.metric("Total Purse", f"${total_purse:,}")
    with col3:
        st.metric("Majors", len(majors))
    with col4:
        st.metric("Tier 1 Events", len(tier1))


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

        with st.spinner("Running simulations..."):
            sim1 = simulator.simulate_tournament(g1, tournament)
            sim2 = simulator.simulate_tournament(g2, tournament)

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
        if plan['elite_reservation']:
            for r in plan['elite_reservation']:
                st.info(f"**{r['tournament']}**: {r['golfer']}")
        else:
            st.caption("No elite reservations")

        # Signature picks
        st.subheader("Signature Event Picks")
        if plan['recommendations']['signatures']:
            data = plan['recommendations']['signatures']
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)

        # Regular picks
        st.subheader("Regular Event Picks")
        if plan['recommendations']['regular']:
            data = plan['recommendations']['regular'][:15]  # Show first 15
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)


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

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Update from API"):
            with st.spinner("Fetching data..."):
                api = st.session_state.api
                api.sync_golfers_to_db()
                preds = api.get_pre_tournament_predictions()
            st.success(f"Synced {len(preds)} predictions!")

    with col2:
        if st.button("Clear Cache"):
            db = st.session_state.db
            db.clear_expired_cache()
            st.success("Cache cleared!")


if __name__ == "__main__":
    main()
