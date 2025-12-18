import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="🏈 NFL Red Zone Analytics Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .success-rate {
        font-size: 32px;
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

plt.style.use("default")
sns.set_palette("husl")

# ========================================
# DATA LOADING (data/ + data/train/)
# ========================================
@st.cache_data
@st.cache_data
def load_all_data():
    """
    Robust loader for:
      <repo_root>/data/supplementary_data.csv
      <repo_root>/data/train/input_2023_w01.csv ... w18
      <repo_root>/data/train/output_2023_w01.csv ... w18
    """
    import pathlib

    # Resolve absolute paths from this app.py location
    app_dir = pathlib.Path(__file__).parent.resolve()
    data_dir = app_dir / "data"
    train_dir = data_dir / "train"

    st.write("app_dir:", app_dir)
    st.write("data_dir exists:", data_dir.exists(), "contents:", list(data_dir.glob("*")))
    st.write("train_dir exists:", train_dir.exists(), "contents:", list(train_dir.glob("*")))

    # ---------- supplementary ----------
    supp_path = data_dir / "supplementary_data.csv"
    try:
        st.info(f"⏳ Loading supplementary data from {supp_path} ...")
        supp_df = pd.read_csv(supp_path)
        st.success(f"✅ Loaded supplementary data: {supp_df.shape}")
    except Exception as e:
        st.error(f"❌ Error loading {supp_path.name}: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ---------- input ----------
    st.info("⏳ Loading input tracking data (18 weeks)...")
    input_dfs = []
    for week in range(1, 19):
        week_str = str(week).zfill(2)
        fpath = train_dir / f"input_2023_w{week_str}.csv"
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                df["week"] = week
                input_dfs.append(df)
            except Exception as e:
                st.warning(f"⚠️ Error loading input week {week}: {e}")
        else:
            st.warning(f"⚠️ Missing input file for week {week}: {fpath}")

    if not input_dfs:
        st.error("❌ No input_2023_wXX.csv files found in data/train/.")
        return supp_df, pd.DataFrame(), pd.DataFrame()

    input_df = pd.concat(input_dfs, ignore_index=True)
    st.success(f"✅ Loaded input data: {input_df.shape}")

    # ---------- output ----------
    st.info("⏳ Loading output tracking data (18 weeks)...")
    output_dfs = []
    for week in range(1, 19):
        week_str = str(week).zfill(2)
        fpath = train_dir / f"output_2023_w{week_str}.csv"
        if fpath.exists():
            try:
                df = pd.read_csv(fpath)
                df["week"] = week
                output_dfs.append(df)
            except Exception as e:
                st.warning(f"⚠️ Error loading output week {week}: {e}")
        else:
            st.warning(f"⚠️ Missing output file for week {week}: {fpath}")

    if not output_dfs:
        st.error("❌ No output_2023_wXX.csv files found in data/train/.")
        return supp_df, input_df, pd.DataFrame()

    output_df = pd.concat(output_dfs, ignore_index=True)
    st.success(f"✅ Loaded output data: {output_df.shape}")

    st.success("🎉 ALL DATA LOADED SUCCESSFULLY!")
    return supp_df, input_df, output_df

# ========================================
# DATA PROCESSING
# ========================================
@st.cache_data
def process_data(supp_df, input_df, output_df):
    merged_df = pd.merge(input_df, supp_df, on=["game_id", "play_id"], how="inner")

    def get_redzone_interval(yardline):
        if 5 <= yardline <= 10:
            return "5-10"
        elif 10 < yardline <= 15:
            return "10-15"
        elif 15 < yardline <= 20:
            return "15-20"
        return None

    merged_df["redzone_interval"] = merged_df["yardline_number"].apply(get_redzone_interval)
    redzone_df = merged_df[merged_df["redzone_interval"].notna()].copy()

    play_summary = redzone_df.groupby(["game_id", "play_id"]).agg(
        redzone_interval=("redzone_interval", "first"),
        possession_team=("possession_team", "first"),
        play_description=("play_description", "first"),
        offense_formation=("offense_formation", "first"),
        receiver_alignment=("receiver_alignment", "first"),
        route_of_targeted_receiver=("route_of_targeted_receiver", "first"),
        pass_result=("pass_result", "first"),
        play_action=("play_action", "first"),
        dropback_type=("dropback_type", "first"),
        team_coverage_man_zone=("team_coverage_man_zone", "first"),
        team_coverage_type=("team_coverage_type", "first"),
        yards_gained=("yards_gained", "first"),
        down=("down", "first"),
        yards_to_go=("yards_to_go", "first"),
    ).reset_index()

    play_summary["is_touchdown"] = play_summary["play_description"].str.contains(
        "TOUCHDOWN", case=False, na=False
    )
    successful_plays = play_summary[play_summary["is_touchdown"]].copy()
    return redzone_df, play_summary, successful_plays

# ========================================
# HELPER FUNCTIONS
# ========================================
def calculate_accel_effort_pct(accel_yd_per_sec2):
    REDZONE_ACCEL_MAX = 2.5
    if accel_yd_per_sec2 is None:
        return None
    try:
        accel_val = float(accel_yd_per_sec2)
    except Exception:
        return None
    raw_ratio = accel_val / REDZONE_ACCEL_MAX if REDZONE_ACCEL_MAX != 0 else 0.0
    mapped = math.atan(raw_ratio) / (math.pi / 2)
    MIN_PCT, MAX_PCT = 75.0, 100.0
    scaled_pct = MIN_PCT + mapped * (MAX_PCT - MIN_PCT)
    return round(scaled_pct, 1)

FEET_TO_YARDS = 3.0

def calculate_receiver_kinematics_with_effort(tracking_data):
    if tracking_data.empty:
        return {"avg_accel": None, "avg_accel_effort_pct": None, "start_x": None, "start_y": None}
    tracking_data = tracking_data.sort_values("frame_id")
    x = tracking_data["x"].values / FEET_TO_YARDS
    y = tracking_data["y"].values / FEET_TO_YARDS
    dt = 0.01
    dx, dy = np.diff(x), np.diff(y)
    dist = np.sqrt(dx**2 + dy**2) if len(dx) > 0 else np.array([])
    v = dist / dt if len(dist) > 0 else np.array([])
    dv = np.diff(v) if len(v) > 1 else np.array([])
    a = dv / dt if len(dv) > 0 else np.array([])
    avg_accel = np.mean(np.abs(a)) if len(a) > 0 else None
    avg_pct = calculate_accel_effort_pct(avg_accel) if avg_accel else None
    return {
        "avg_accel": round(avg_accel, 2) if avg_accel else None,
        "avg_accel_effort_pct": avg_pct,
        "start_x": x[0] if len(x) > 0 else None,
        "start_y": y[0] if len(y) > 0 else None,
    }

def simulate_play_reliability(successes, attempts, play_summary, successful_plays, global_avg_rate=None, simulations=10000):
    if global_avg_rate is None:
        if len(play_summary) > 0:
            global_avg_rate = max(len(successful_plays) / len(play_summary), 0.15)
        else:
            global_avg_rate = 0.15
    prior_strength = 8
    prior_alpha = max(global_avg_rate * prior_strength, 1)
    prior_beta = max((1 - global_avg_rate) * prior_strength, 1)
    post_a = prior_alpha + successes
    post_b = prior_beta + (attempts - successes)
    sims = stats.beta.rvs(post_a, post_b, size=simulations)
    return round(np.mean(sims) * 100, 1), round(np.percentile(sims, 25) * 100, 1)

ROUTE_ACCELERATION_MAP = {
    "post": 80,
    "go": 100,
    "cross": 70,
    "corner": 80,
    "wheel": 80,
    "angle": 60,
    "flat": 90,
}

def get_route_acceleration_pct(route_name):
    if pd.isna(route_name):
        return None
    return ROUTE_ACCELERATION_MAP.get(str(route_name).lower().strip(), 75)

def map_coverage_input(user_input, available_covs):
    user_input = user_input.strip().upper()
    mapping = {
        "MAN": "MAN_COVERAGE",
        "ZONE": "ZONE_COVERAGE",
        "MAN_COVERAGE": "MAN_COVERAGE",
        "ZONE_COVERAGE": "ZONE_COVERAGE",
    }
    mapped = mapping.get(user_input, user_input)
    if mapped in available_covs:
        return mapped
    for cov in available_covs:
        if user_input in cov.upper():
            return cov
    return available_covs[0] if len(available_covs) > 0 else user_input

def get_enhanced_recommendations_final(yards_out, defense_type, play_summary, successful_plays, redzone_df):
    if 5 <= yards_out <= 10:
        interval = "5-10"
    elif 10 < yards_out <= 15:
        interval = "10-15"
    elif 15 < yards_out <= 20:
        interval = "15-20"
    else:
        return {"error": "Yards must be between 5-20"}

    available_covs = play_summary["team_coverage_man_zone"].unique()
    mapped_coverage = map_coverage_input(defense_type, available_covs)

    scenario_plays = successful_plays[
        (successful_plays["redzone_interval"] == interval)
        & (successful_plays["team_coverage_man_zone"] == mapped_coverage)
    ]
    all_attempts = play_summary[
        (play_summary["redzone_interval"] == interval)
        & (play_summary["team_coverage_man_zone"] == mapped_coverage)
    ]
    if len(all_attempts) < 5:
        return {"error": f"Insufficient sample size ({len(all_attempts)} plays)."}

    grouped = all_attempts.groupby(
        ["offense_formation", "route_of_targeted_receiver", "receiver_alignment"]
    ).agg(
        total_attempts=("play_id", "count"),
        td_count=("is_touchdown", "sum"),
    ).reset_index()
    grouped = grouped[grouped["total_attempts"] >= 1]
    if grouped.empty:
        return {"error": "No patterns found."}

    results = []
    scenario_avg = len(scenario_plays) / max(len(all_attempts), 1)

    for _, row in grouped.iterrows():
        exp_rate, rel_score = simulate_play_reliability(
            row["td_count"], row["total_attempts"], play_summary, successful_plays, global_avg_rate=scenario_avg
        )
        raw_rate = (row["td_count"] / row["total_attempts"]) * 100

        td_plays = scenario_plays[
            (scenario_plays["offense_formation"] == row["offense_formation"])
            & (scenario_plays["route_of_targeted_receiver"] == row["route_of_targeted_receiver"])
            & (scenario_plays["receiver_alignment"] == row["receiver_alignment"])
        ]
        play_ids = td_plays["play_id"].values
        pattern_tracking = redzone_df[redzone_df["play_id"].isin(play_ids)]

        receiver_stats = []
        if "player_to_predict" in pattern_tracking.columns:
            for pid in play_ids:
                pt = pattern_tracking[pattern_tracking["play_id"] == pid]
                for p in pt["player_to_predict"].unique():
                    ptrack = pt[pt["player_to_predict"] == p]
                    if len(ptrack) > 1:
                        receiver_stats.append(
                            calculate_receiver_kinematics_with_effort(ptrack)
                        )

        route_accel_pct = get_route_acceleration_pct(row["route_of_targeted_receiver"])

        if receiver_stats:
            sx = [r["start_x"] for r in receiver_stats if r["start_x"] is not None]
            sy = [r["start_y"] for r in receiver_stats if r["start_y"] is not None]
            start_x = np.mean(sx) if sx else None
            start_y = np.mean(sy) if sy else None
            pos_fx = np.std(sx) if len(sx) > 1 else None
            pos_fy = np.std(sy) if len(sy) > 1 else None
        else:
            start_x = start_y = pos_fx = pos_fy = None

        results.append(
            {
                "formation": row["offense_formation"],
                "route": row["route_of_targeted_receiver"],
                "alignment": row["receiver_alignment"],
                "td_count": row["td_count"],
                "attempts": row["total_attempts"],
                "raw_success_rate": round(raw_rate, 1),
                "simulated_success": exp_rate,
                "reliability_score": rel_score,
                "avg_accel_effort_pct": route_accel_pct,
                "start_x": round(start_x, 1) if start_x else None,
                "start_y": round(start_y, 1) if start_y else None,
                "pos_flex_x": round(pos_fx, 2) if pos_fx else None,
                "pos_flex_y": round(pos_fy, 2) if pos_fy else None,
            }
        )

    df_results = pd.DataFrame(results).sort_values("reliability_score", ascending=False)
    return {
        "scenario": {"interval": interval, "defense": mapped_coverage},
        "data": df_results.head(5).to_dict("records"),
    }

# ========================================
# MAIN APP
# ========================================
st.title("🏈 NFL Red Zone Analytics Dashboard")
st.markdown("*Defender Distance & Separation Strategy Analysis*")
st.markdown("---")

with st.spinner("Loading NFL data (this may take a moment)..."):
    supp_df, input_df, output_df = load_all_data()
    if supp_df.empty or input_df.empty or output_df.empty:
        st.stop()
    redzone_df, play_summary, successful_plays = process_data(supp_df, input_df, output_df)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("📊 Total Plays", len(play_summary))
with c2:
    st.metric("🏈 Touchdown Plays", len(successful_plays))
with c3:
    td_rate = len(successful_plays) / len(play_summary) * 100 if len(play_summary) > 0 else 0
    st.metric("✅ TD Success Rate", f"{td_rate:.1f}%")

st.markdown("---")

st.sidebar.header("🎛️ Analysis Parameters")
yards_out = st.sidebar.slider("Distance from Endzone (yards)", 5, 20, 10, 1)
available_covs = sorted(play_summary["team_coverage_man_zone"].dropna().unique().tolist())
defense_type = st.sidebar.selectbox("Defense Coverage Type", available_covs)

if st.sidebar.button("🔍 Analyze Play Patterns", use_container_width=True):
    st.session_state.analyze = True
else:
    st.session_state.analyze = False

if st.session_state.get("analyze"):
    with st.spinner("🔄 Analyzing play patterns..."):
        rec = get_enhanced_recommendations_final(
            yards_out, defense_type, play_summary, successful_plays, redzone_df
        )

    if "error" not in rec:
        st.success(
            f"✅ Analysis Complete: {rec['scenario']['interval']} yards vs {rec['scenario']['defense']}"
        )
        df_display = pd.DataFrame(rec["data"]).rename(
            columns={
                "formation": "Formation",
                "route": "Route",
                "alignment": "Alignment",
                "td_count": "TDs",
                "attempts": "Attempts",
                "raw_success_rate": "Raw Rate %",
                "simulated_success": "Simulated Success %",
                "reliability_score": "Reliability %",
            }
        )
        st.subheader("📋 Top Play Patterns")
        st.dataframe(
            df_display[
                [
                    "Formation",
                    "Route",
                    "Alignment",
                    "TDs",
                    "Attempts",
                    "Raw Rate %",
                    "Simulated Success %",
                    "Reliability %",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            dfs = df_display.sort_values("Simulated Success %")
            ax.barh(range(len(dfs)), dfs["Simulated Success %"], color="#1f77b4")
            ax.set_yticks(range(len(dfs)))
            ax.set_yticklabels(
                [f"{r['Route']} - {r['Formation']}" for _, r in dfs.iterrows()]
            )
            ax.set_xlabel("Success Rate (%)")
            ax.set_title("Success Rates by Play Pattern")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            dfs = df_display.sort_values("Reliability %")
            ax.barh(range(len(dfs)), dfs["Reliability %"], color="#2ca02c")
            ax.set_yticks(range(len(dfs)))
            ax.set_yticklabels(
                [f"{r['Route']} - {r['Formation']}" for _, r in dfs.iterrows()]
            )
            ax.set_xlabel("Reliability Score (%)")
            ax.set_title("Reliability by Play Pattern")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.error(f"❌ {rec['error']}")

st.markdown("---")
st.markdown("""
    ### 📌 About This Dashboard
    This dashboard analyzes NFL Red Zone play-calling strategies by examining:
    - **Defender Distance Impact**
    - **Coverage Effects**
    - **Route-Formation Combinations**
    - **Bayesian Reliability**
""")
