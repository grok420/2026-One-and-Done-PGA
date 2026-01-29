"""
PGA One and Done Optimizer - Streamlined Web App v2
Focus: Clear "why" explanations + better visualizations

Run with: streamlit run web_app_v2.py
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
from models import Tier, CutRule

# Page config
st.set_page_config(
    page_title="PGA One and Done Optimizer",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    /* Header styling */
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1a472a 0%, #2d5a3d 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }

    /* Tournament info card */
    .tournament-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
    }

    /* Top pick card styling */
    .top-pick-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        border-left: 6px solid #FFD700;
    }

    /* Why section */
    .why-section {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }

    /* Risk flags */
    .risk-flag {
        background: #dc3545;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.85rem;
        margin-right: 0.5rem;
    }

    /* Timing badges */
    .timing-now { background: #28a745; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; }
    .timing-save { background: #ffc107; color: black; padding: 0.25rem 0.75rem; border-radius: 20px; }
    .timing-tossup { background: #6c757d; color: white; padding: 0.25rem 0.75rem; border-radius: 20px; }

    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'db' not in st.session_state:
        st.session_state.db = Database()
    if 'strategy' not in st.session_state:
        st.session_state.strategy = Strategy()
    if 'api' not in st.session_state:
        st.session_state.api = DataGolfAPI()

    # Auto-sync on first load if database is empty
    if 'data_synced' not in st.session_state:
        db = st.session_state.db
        api = st.session_state.api
        if db.get_golfer_count() == 0 or db.get_valid_owgr_count() == 0:
            with st.spinner("First load - syncing golfer data from Data Golf API..."):
                try:
                    api.sync_golfers_to_db()
                    st.session_state.data_synced = True
                except Exception as e:
                    st.warning(f"Auto-sync failed: {e}. Go to Settings to sync manually.")
        else:
            st.session_state.data_synced = True


def create_waterfall_chart(rec):
    """Create factor contribution waterfall chart."""
    if not rec.factor_contributions:
        return None

    factors = rec.factor_contributions

    # Build waterfall data
    labels = ["Base EV"]
    values = [rec.base_ev]
    measure = ["absolute"]

    factor_names = {
        "course_fit": "Course Fit",
        "timing": "Timing",
        "field_strength": "Field",
        "hedge": "Hedge",
        "phase": "Season Phase",
        "course_history": "History"
    }

    for key, value in factors.items():
        if abs(value) > 1000:  # Only show significant factors
            labels.append(factor_names.get(key, key))
            values.append(value)
            measure.append("relative")

    labels.append("Final EV")
    values.append(rec.expected_value)
    measure.append("total")

    fig = go.Figure(go.Waterfall(
        orientation="v",
        measure=measure,
        x=labels,
        y=values,
        textposition="outside",
        text=[f"${v/1000:.0f}K" for v in values],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#28a745"}},
        decreasing={"marker": {"color": "#dc3545"}},
        totals={"marker": {"color": "#1e3c72"}}
    ))

    fig.update_layout(
        title="Factor Breakdown",
        showlegend=False,
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def create_bubble_chart(recs):
    """Create win% vs EV bubble chart."""
    if not recs:
        return None

    data = []
    for rec in recs[:15]:
        timing_color = "green" if "NOW" in rec.timing_verdict else "orange" if "SAVE" in rec.timing_verdict else "gray"
        data.append({
            "Golfer": rec.golfer.name,
            "Win %": rec.golfer.win_probability * 100,
            "Expected Value": rec.expected_value,
            "Confidence": rec.confidence_pct,
            "Timing": rec.timing_verdict,
            "Color": timing_color
        })

    df = pd.DataFrame(data)

    fig = px.scatter(
        df,
        x="Win %",
        y="Expected Value",
        size="Confidence",
        color="Timing",
        hover_name="Golfer",
        color_discrete_map={
            "USE NOW": "#28a745",
            "TOSS-UP": "#6c757d",
        },
        size_max=40,
    )

    # Color SAVE recommendations orange
    for trace in fig.data:
        if "SAVE" in trace.name:
            trace.marker.color = "#ffc107"

    fig.update_layout(
        title="Pick Comparison: Win% vs Expected Value",
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )

    return fig


def render_top_pick_card(rec):
    """Render the top pick recommendation card."""
    g = rec.golfer

    # Timing badge HTML
    timing_class = "timing-now" if "NOW" in rec.timing_verdict else "timing-save" if "SAVE" in rec.timing_verdict else "timing-tossup"

    st.markdown(f"""
    <div class="top-pick-card">
        <h2 style="margin:0;">🏆 #1 PICK: {g.name}</h2>
        <div style="font-size: 1.5rem; margin: 0.5rem 0;">
            Expected Value: <strong>${rec.expected_value:,.0f}</strong> |
            Win: <strong>{g.win_probability*100:.1f}%</strong> |
            Top-10: <strong>{g.top_10_probability*100:.1f}%</strong>
        </div>
        <div>
            <span class="{timing_class}">{rec.timing_verdict}</span>
            <span style="margin-left: 1rem;">Confidence: {rec.confidence_pct}%</span>
            <span style="margin-left: 1rem;">OWGR: #{g.owgr}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_why_section(rec):
    """Render the WHY THIS PICK section."""
    st.markdown("### 💡 WHY THIS PICK?")

    if rec.plain_english_bullets:
        for bullet in rec.plain_english_bullets:
            st.markdown(f"✓ {bullet}")
    else:
        # Fallback to generating bullets from available data
        st.markdown(f"✓ Win probability: {rec.golfer.win_probability*100:.1f}%")
        st.markdown(f"✓ Top-10 probability: {rec.golfer.top_10_probability*100:.1f}%")
        if rec.course_fit_sg > 0:
            st.markdown(f"✓ Course fit: +{rec.course_fit_sg:.2f} SG/round")

    # Risk flags
    if rec.risk_flags:
        st.markdown("#### ⚠️ Risk Factors")
        for flag in rec.risk_flags:
            st.markdown(f"- {flag}")


def render_other_picks(recs):
    """Render the other top recommendations grid."""
    st.markdown("### 📊 Other Top Picks")

    # Create comparison dataframe
    data = []
    for i, rec in enumerate(recs[1:6], 2):  # Picks 2-6
        g = rec.golfer
        data.append({
            "Rank": f"#{i}",
            "Golfer": g.name,
            "EV": f"${rec.expected_value:,.0f}",
            "Win%": f"{g.win_probability*100:.1f}%",
            "Top-10%": f"{g.top_10_probability*100:.1f}%",
            "OWGR": g.owgr,
            "Timing": rec.timing_verdict,
            "Confidence": f"{rec.confidence_pct}%",
        })

    df = pd.DataFrame(data)

    # Style the dataframe
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Rank": st.column_config.TextColumn("Rank", width="small"),
            "Golfer": st.column_config.TextColumn("Golfer", width="medium"),
            "EV": st.column_config.TextColumn("Expected Value", width="small"),
            "Win%": st.column_config.TextColumn("Win%", width="small"),
            "Top-10%": st.column_config.TextColumn("Top-10%", width="small"),
            "OWGR": st.column_config.NumberColumn("OWGR", width="small"),
            "Timing": st.column_config.TextColumn("Timing", width="medium"),
            "Confidence": st.column_config.TextColumn("Conf.", width="small"),
        }
    )


def render_pick_detail_expander(rec, rank):
    """Render detailed pick info in an expander."""
    g = rec.golfer

    with st.expander(f"#{rank} {g.name} - ${rec.expected_value:,.0f} EV"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Probabilities**")
            st.markdown(f"- Win: {g.win_probability*100:.1f}%")
            st.markdown(f"- Top-10: {g.top_10_probability*100:.1f}%")
            st.markdown(f"- Top-20: {g.top_20_probability*100:.1f}%")
            st.markdown(f"- Make Cut: {g.make_cut_probability*100:.1f}%")

        with col2:
            st.markdown("**Analysis**")
            st.markdown(f"- OWGR: #{g.owgr}")
            st.markdown(f"- Course Fit: {'+' if rec.course_fit_sg > 0 else ''}{rec.course_fit_sg:.2f} SG/rd")
            st.markdown(f"- Timing: {rec.timing_verdict}")
            st.markdown(f"- Confidence: {rec.confidence_pct}%")

        if rec.plain_english_bullets:
            st.markdown("**Why this pick?**")
            for bullet in rec.plain_english_bullets:
                st.markdown(f"- {bullet}")


def page_this_week():
    """This Week page - main decision interface."""
    strategy = st.session_state.strategy
    db = st.session_state.db

    # Get next tournament
    tournament = get_next_tournament()
    if not tournament:
        st.error("No upcoming tournament found")
        return

    # Tournament info header
    tier_badge = "🏆 MAJOR" if tournament.is_major else "⭐ SIGNATURE" if tournament.is_signature else ""
    cut_info = "No Cut" if not tournament.has_cut else f"Cut: {tournament.cut_rule.value}"

    st.markdown(f"""
    <div class="tournament-card">
        <h2 style="margin:0;">{tournament.name} {tier_badge}</h2>
        <div style="font-size: 1.1rem; margin-top: 0.5rem;">
            📅 {tournament.date.strftime('%B %d, %Y')} |
            ⛳ {tournament.course} |
            💰 ${tournament.purse:,} Purse |
            {cut_info}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # User status row
    standings = db.get_latest_standings()
    used_golfers = db.get_used_golfers()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        # Find user in standings (simplified - just show picks used)
        st.metric("Picks Used", len(used_golfers))
    with col2:
        st.metric("Purse", f"${tournament.purse/1_000_000:.1f}M")
    with col3:
        st.metric("Winner Share", f"${tournament.winner_share/1_000_000:.2f}M")
    with col4:
        st.metric("Tier", tournament.tier.name)

    st.divider()

    # Get recommendations
    with st.spinner("Analyzing field and generating recommendations..."):
        try:
            recs = strategy.get_recommendations(tournament, top_n=15)
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            st.info("Try syncing data from the Settings page")
            return

    if not recs:
        st.warning("No recommendations available. Make sure golfer data is synced.")
        return

    # Top Pick Card
    top_rec = recs[0]
    render_top_pick_card(top_rec)

    # Two column layout: Why section + Waterfall chart
    col1, col2 = st.columns([1, 1])

    with col1:
        render_why_section(top_rec)

    with col2:
        waterfall = create_waterfall_chart(top_rec)
        if waterfall:
            st.plotly_chart(waterfall, use_container_width=True)

    st.divider()

    # Other picks table
    render_other_picks(recs)

    # Bubble chart comparison
    st.markdown("### 🎯 Visual Comparison")
    bubble = create_bubble_chart(recs)
    if bubble:
        st.plotly_chart(bubble, use_container_width=True)

    # Detailed pick expanders
    st.markdown("### 📋 Detailed Analysis")
    for i, rec in enumerate(recs[:10], 1):
        render_pick_detail_expander(rec, i)


def page_season_plan():
    """Season Plan page - resource allocation."""
    st.markdown("## 📅 Season Plan")

    db = st.session_state.db
    schedule = get_schedule()
    today = date.today()

    # Resource summary
    used_golfers = db.get_used_golfers()
    all_golfers = db.get_all_golfers()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Golfers Used", len(used_golfers))
    with col2:
        remaining = len([t for t in schedule if t.date >= today])
        st.metric("Tournaments Remaining", remaining)
    with col3:
        elites_available = len([g for g in all_golfers if g.owgr <= 20 and g.name not in used_golfers])
        st.metric("Elites Available", elites_available)

    st.divider()

    # Upcoming tournaments
    st.markdown("### Upcoming Tournaments")

    upcoming = [t for t in schedule if t.date >= today][:10]

    data = []
    for t in upcoming:
        tier_icon = "🏆" if t.is_major else "⭐" if t.is_signature else ""
        data.append({
            "Date": t.date.strftime("%b %d"),
            "Tournament": f"{t.name} {tier_icon}",
            "Course": t.course,
            "Purse": f"${t.purse/1_000_000:.1f}M",
            "Tier": t.tier.name,
            "Cut": "No" if not t.has_cut else "Yes",
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Used golfers list
    st.markdown("### Used Golfers")
    if used_golfers:
        cols = st.columns(4)
        for i, golfer in enumerate(used_golfers):
            with cols[i % 4]:
                st.markdown(f"• {golfer}")
    else:
        st.info("No golfers used yet")


def page_insights():
    """Insights page - performance tracking."""
    st.markdown("## 📊 Insights")

    db = st.session_state.db

    # League standings
    standings = db.get_latest_standings()

    if standings:
        st.markdown("### League Standings")

        data = []
        for s in standings[:20]:
            data.append({
                "Rank": s.rank,
                "Player": s.player_name,
                "Earnings": f"${s.total_earnings:,}",
                "Cuts Made": s.cuts_made,
                "Picks": s.picks_made,
            })

        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No standings data available. Sync from Settings.")


def page_schedule():
    """Schedule page - full season calendar."""
    st.markdown("## 📅 2026 Tournament Schedule")

    schedule = get_schedule()
    today = date.today()

    # Filter options
    filter_type = st.selectbox(
        "Filter",
        ["All Tournaments", "Majors Only", "Signature Events", "Remaining Only"]
    )

    if filter_type == "Majors Only":
        schedule = [t for t in schedule if t.is_major]
    elif filter_type == "Signature Events":
        schedule = [t for t in schedule if t.is_signature]
    elif filter_type == "Remaining Only":
        schedule = [t for t in schedule if t.date >= today]

    data = []
    for t in schedule:
        status = "✓ Complete" if t.date < today else "Upcoming"
        tier_icon = "🏆" if t.is_major else "⭐" if t.is_signature else ""
        data.append({
            "Date": t.date.strftime("%b %d, %Y"),
            "Tournament": f"{t.name} {tier_icon}",
            "Course": t.course,
            "Purse": f"${t.purse/1_000_000:.1f}M",
            "Tier": t.tier.name,
            "Status": status,
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def page_settings():
    """Settings page."""
    st.markdown("## ⚙️ Settings")

    db = st.session_state.db
    api = st.session_state.api

    # Data sync section
    st.markdown("### Data Sync")

    col1, col2 = st.columns(2)
    with col1:
        golfer_count = db.get_golfer_count()
        st.metric("Golfers in Database", golfer_count)
    with col2:
        valid_owgr = db.get_valid_owgr_count()
        st.metric("With Valid OWGR", valid_owgr)

    if st.button("🔄 Sync Data from Data Golf API", type="primary"):
        with st.spinner("Syncing..."):
            try:
                api.sync_golfers_to_db()
                st.success("Data synced successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Sync failed: {e}")

    st.divider()

    # API status
    st.markdown("### API Status")
    config = get_config()

    if config.datagolf_api_key:
        st.success("Data Golf API key configured")
    else:
        st.warning("Data Golf API key not set. Add DATAGOLF_API_KEY to .env file.")


def main():
    """Main application."""
    init_session_state()

    # Header
    st.markdown('<div class="main-header">⛳ PGA One and Done Optimizer</div>', unsafe_allow_html=True)

    # Navigation tabs
    tabs = st.tabs(["🎯 This Week", "📅 Season Plan", "📊 Insights", "📆 Schedule", "⚙️ Settings"])

    with tabs[0]:
        page_this_week()

    with tabs[1]:
        page_season_plan()

    with tabs[2]:
        page_insights()

    with tabs[3]:
        page_schedule()

    with tabs[4]:
        page_settings()


if __name__ == "__main__":
    main()
