import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
import os
from pathlib import Path
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

# ========================================
# STYLING
# ========================================
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

plt.style.use('default')
sns.set_palette("husl")

# ========================================
# DATA LOADING (CACHED) - FIXED VERSION
# ========================================
@st.cache_data
def load_all_data():
    """
    Load supplementary, input (18 weeks), and output (18 weeks) data
    from a local data folder or Kaggle folder structure.
    """
    import os
    
    # Debug: show what Streamlit sees
    st.write("🔍 Debug - CWD:", os.getcwd())
    st.write("🔍 Debug - Root files:", os.listdir())
    
    # Try to find data folder
    base_path = None
    
    # Option 1: Look for data/ folder (preferred)
    if os.path.exists("data"):
        base_path = "data"
        st.write("✅ Found data/ folder")
        st.write("🔍 Debug - Files in data/:", os.listdir("data"))
    
    # Option 2: Look for Kaggle folder structure
    elif os.path.exists("114239_nfl_competition_files_published_analytics_final"):
        base_path = "114239_nfl_competition_files_published_analytics_final"
        st.write("✅ Found Kaggle folder structure")
    
    else:
        st.error("❌ No data folder found. Expected either 'data/' or '114239_nfl_competition_files_published_analytics_final/'")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # ---------- supplementary ----------
    supp_path = os.path.join(base_path, "supplementary_data.csv")
    st.info(f"⏳ Loading supplementary data from {supp_path} ...")
    
    try:
        supp_df = pd.read_csv(supp_path)
        st.success(f"✅ Loaded supplementary data: {supp_df.shape}")
    except FileNotFoundError as e:
        st.error(f"❌ supplementary_data.csv not found: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading supplementary_data.csv: {e}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # ---------- input (18 weeks) ----------
    st.info("⏳ Loading input tracking data (18 weeks)...")
    input_dfs = []
    
    # Determine if we need to look in train/ subfolder
    train_folder = base_path
    if base_path == "114239_nfl_competition_files_published_analytics_final":
        train_folder = os.path.join(base_path, "train")
    
    for week in range(1, 19):
        week_str = str(week).zfill(2)
        input_file = os.path.join(train_folder, f"input_2023_w{week_str}.csv")
        
        if os.path.exists(input_file):
            try:
                df_week = pd.read_csv(input_file)
                df_week["week"] = week
                input_dfs.append(df_week)
            except Exception as e:
                st.warning(f"⚠️ Error loading input week {week}: {e}")
        else:
            st.warning(f"⚠️ Missing input file for week {week}: {input_file}")

    if not input_dfs:
        st.error("❌ No input_2023_wXX.csv files found.")
        return supp_df, pd.DataFrame(), pd.DataFrame()

    input_df = pd.concat(input_dfs, ignore_index=True)
    st.success(f"✅ Loaded input data (all weeks): {input_df.shape}")

    # ---------- output (18 weeks) ----------
    st.info("⏳ Loading output tracking data (18 weeks)...")
    output_dfs = []
    
    for week in range(1, 19):
        week_str = str(week).zfill(2)
        output_file = os.path.join(train_folder, f"output_2023_w{week_str}.csv")
        
        if os.path.exists(output_file):
            try:
                df_week = pd.read_csv(output_file)
                df_week["week"] = week
                output_dfs.append(df_week)
            except Exception as e:
                st.warning(f"⚠️ Error loading output week {week}: {e}")
        else:
            st.warning(f"⚠️ Missing output file for week {week}: {output_file}")

    if not output_dfs:
        st.error("❌ No output_2023_wXX.csv files found.")
        return supp_df, input_df, pd.DataFrame()

    output_df = pd.concat(output_dfs, ignore_index=True)
    st.success(f"✅ Loaded output data (all weeks): {output_df.shape}")

    st.success("🎉 ALL DATA LOADED SUCCESSFULLY!")
    
    return supp_df, input_df, output_df

# ========================================
# DATA PROCESSING
# ========================================
@st.cache_data
def process_data(supp_df, input_df, output_df):
    """Process and merge all data"""
    
    # Merge tracking data with play context
    merged_df = pd.merge(input_df, supp_df, on=['game_id', 'play_id'], how='inner')
    
    # Define redzone intervals
    def get_redzone_interval(yardline):
        if 5 <= yardline <= 10:
            return '5-10'
        elif 10 < yardline <= 15:
            return '10-15'
        elif 15 < yardline <= 20:
            return '15-20'
        return None
    
    merged_df['redzone_interval'] = merged_df['yardline_number'].apply(get_redzone_interval)
    redzone_df = merged_df[merged_df['redzone_interval'].notna()].copy()
    
    # Create play summary
    play_summary = redzone_df.groupby(['game_id', 'play_id']).agg({
        'redzone_interval': 'first',
        'possession_team': 'first',
        'play_description': 'first',
        'offense_formation': 'first',
        'receiver_alignment': 'first',
        'route_of_targeted_receiver': 'first',
        'pass_result': 'first',
        'play_action': 'first',
        'dropback_type': 'first',
        'team_coverage_man_zone': 'first',
        'team_coverage_type': 'first',
        'yards_gained': 'first',
        'down': 'first',
        'yards_to_go': 'first'
    }).reset_index()
    
    # Identify touchdown plays
    play_summary['is_touchdown'] = play_summary['play_description'].str.contains('TOUCHDOWN', case=False, na=False)
    successful_plays = play_summary[play_summary['is_touchdown'] == True].copy()
    
    return redzone_df, play_summary, successful_plays

# ========================================
# HELPER FUNCTIONS
# ========================================

def calculate_accel_effort_pct(accel_yd_per_sec2):
    """Calculate acceleration effort percentage"""
    REDZONE_ACCEL_MAX = 2.5
    if accel_yd_per_sec2 is None:
        return None
    try:
        accel_val = float(accel_yd_per_sec2)
    except Exception:
        return None
    
    raw_ratio = accel_val / REDZONE_ACCEL_MAX if REDZONE_ACCEL_MAX != 0 else 0.0
    mapped = math.atan(raw_ratio) / (math.pi / 2)
    MIN_PCT = 75.0
    MAX_PCT = 100.0
    scaled_pct = MIN_PCT + mapped * (MAX_PCT - MIN_PCT)
    return round(scaled_pct, 1)

FEET_TO_YARDS = 3.0

def calculate_receiver_kinematics_with_effort(tracking_data):
    """Calculate receiver kinematics"""
    if tracking_data.empty:
        return {
            'avg_accel': None,
            'avg_accel_effort_pct': None,
            'start_x': None,
            'start_y': None,
        }
    
    tracking_data = tracking_data.sort_values('frame_id')
    x = tracking_data['x'].values / FEET_TO_YARDS
    y = tracking_data['y'].values / FEET_TO_YARDS
    
    frame_interval = 0.01
    
    dx = np.diff(x)
    dy = np.diff(y)
    distance_per_frame = np.sqrt(dx**2 + dy**2) if len(dx) > 0 else np.array([])
    
    speed_per_frame = distance_per_frame / frame_interval if len(distance_per_frame) > 0 else np.array([])
    dv = np.diff(speed_per_frame) if len(speed_per_frame) > 1 else np.array([])
    acceleration_per_frame = dv / frame_interval if len(dv) > 0 else np.array([])
    
    avg_accel = np.mean(np.abs(acceleration_per_frame)) if len(acceleration_per_frame) > 0 else None
    avg_accel_effort_pct = calculate_accel_effort_pct(avg_accel) if avg_accel else None
    
    start_x = x[0] if len(x) > 0 else None
    start_y = y[0] if len(y) > 0 else None
    
    return {
        'avg_accel': round(avg_accel, 2) if avg_accel else None,
        'avg_accel_effort_pct': avg_accel_effort_pct,
        'start_x': start_x,
        'start_y': start_y,
    }

def simulate_play_reliability(successes, attempts, play_summary, successful_plays, global_avg_rate=None, simulations=10000):
    """Simulate play success using Bayesian statistics"""
    if global_avg_rate is None:
        if len(play_summary) > 0:
            global_avg_rate = max(len(successful_plays) / len(play_summary), 0.15)
        else:
            global_avg_rate = 0.15
    
    prior_strength = 8
    prior_alpha = max(global_avg_rate * prior_strength, 1)
    prior_beta = max((1 - global_avg_rate) * prior_strength, 1)
    
    posterior_alpha = prior_alpha + successes
    posterior_beta = prior_beta + (attempts - successes)
    
    simulated_rates = stats.beta.rvs(posterior_alpha, posterior_beta, size=simulations)
    
    expected_rate = np.mean(simulated_rates) * 100
    lower_bound_25 = np.percentile(simulated_rates, 25) * 100
    
    return round(expected_rate, 1), round(lower_bound_25, 1)

ROUTE_ACCELERATION_MAP = {
    'post': 80,
    'go': 100,
    'cross': 70,
    'corner': 80,
    'wheel': 80,
    'angle': 60,
    'flat': 90,
}

def get_route_acceleration_pct(route_name):
    """Get route acceleration percentage"""
    if pd.isna(route_name):
        return None
    route_lower = str(route_name).lower().strip()
    return ROUTE_ACCELERATION_MAP.get(route_lower, 75)

def map_coverage_input(user_input, available_covs):
    """Map user input to coverage type"""
    user_input = user_input.strip().upper()
    mapping = {
        'MAN': 'MAN_COVERAGE',
        'ZONE': 'ZONE_COVERAGE',
        'MAN_COVERAGE': 'MAN_COVERAGE',
        'ZONE_COVERAGE': 'ZONE_COVERAGE',
    }
    mapped = mapping.get(user_input, user_input)
    if mapped in available_covs:
        return mapped
    for cov in available_covs:
        if user_input in cov.upper():
            return cov
    return available_covs[0] if len(available_covs) > 0 else user_input

def get_enhanced_recommendations_final(yards_out, defense_type, play_summary, successful_plays, redzone_df):
    """Get play recommendations"""
    if 5 <= yards_out <= 10:
        interval = '5-10'
    elif 10 < yards_out <= 15:
        interval = '10-15'
    elif 15 < yards_out <= 20:
        interval = '15-20'
    else:
        return {'error': 'Yards must be between 5-20'}
    
    available_covs = play_summary['team_coverage_man_zone'].unique()
    mapped_coverage = map_coverage_input(defense_type, available_covs)
    
    scenario_plays = successful_plays[
        (successful_plays['redzone_interval'] == interval) &
        (successful_plays['team_coverage_man_zone'] == mapped_coverage)
    ]
    
    all_attempts_scenario = play_summary[
        (play_summary['redzone_interval'] == interval) &
        (play_summary['team_coverage_man_zone'] == mapped_coverage)
    ]
    
    if len(all_attempts_scenario) < 5:
        return {'error': f"Insufficient sample size ({len(all_attempts_scenario)} plays)."}
    
    grouped = all_attempts_scenario.groupby([
        'offense_formation',
        'route_of_targeted_receiver',
        'receiver_alignment'
    ]).agg(
        total_attempts=('play_id', 'count'),
        td_count=('is_touchdown', 'sum')
    ).reset_index()
    
    grouped = grouped[grouped['total_attempts'] >= 1].copy()
    
    if grouped.empty:
        return {'error': "No patterns found."}
    
    results = []
    scenario_avg = len(scenario_plays) / max(len(all_attempts_scenario), 1)
    
    for _, row in grouped.iterrows():
        expected_rate, reliability_score = simulate_play_reliability(
            row['td_count'],
            row['total_attempts'],
            play_summary,
            successful_plays,
            global_avg_rate=scenario_avg
        )
        
        raw_rate = (row['td_count'] / row['total_attempts']) * 100
        
        td_plays = scenario_plays[
            (scenario_plays['offense_formation'] == row['offense_formation']) &
            (scenario_plays['route_of_targeted_receiver'] == row['route_of_targeted_receiver']) &
            (scenario_plays['receiver_alignment'] == row['receiver_alignment'])
        ]
        
        play_ids = td_plays['play_id'].values
        pattern_tracking = redzone_df[redzone_df['play_id'].isin(play_ids)]
        
        receiver_stats_per_play = []
        
        for play_id in play_ids:
            play_tracking = pattern_tracking[pattern_tracking['play_id'] == play_id]
            if 'player_to_predict' in play_tracking.columns:
                for player in play_tracking['player_to_predict'].unique():
                    player_tracking = play_tracking[play_tracking['player_to_predict'] == player]
                    if len(player_tracking) > 1:
                        kinematics = calculate_receiver_kinematics_with_effort(player_tracking)
                        receiver_stats_per_play.append(kinematics)
        
        route_accel_pct = get_route_acceleration_pct(row['route_of_targeted_receiver'])
        
        if receiver_stats_per_play:
            start_x_vals = [r['start_x'] for r in receiver_stats_per_play if r['start_x'] is not None]
            start_y_vals = [r['start_y'] for r in receiver_stats_per_play if r['start_y'] is not None]
            
            start_x = np.mean(start_x_vals) if start_x_vals else None
            start_y = np.mean(start_y_vals) if start_y_vals else None
            pos_flex_x = np.std(start_x_vals) if len(start_x_vals) > 1 else None
            pos_flex_y = np.std(start_y_vals) if len(start_y_vals) > 1 else None
        else:
            start_x = start_y = pos_flex_x = pos_flex_y = None
        
        results.append({
            'formation': row['offense_formation'],
            'route': row['route_of_targeted_receiver'],
            'alignment': row['receiver_alignment'],
            'td_count': row['td_count'],
            'attempts': row['total_attempts'],
            'raw_success_rate': round(raw_rate, 1),
            'simulated_success': expected_rate,
            'reliability_score': reliability_score,
            'avg_accel_effort_pct': route_accel_pct,
            'start_x': round(start_x, 1) if start_x else None,
            'start_y': round(start_y, 1) if start_y else None,
            'pos_flex_x': round(pos_flex_x, 2) if pos_flex_x else None,
            'pos_flex_y': round(pos_flex_y, 2) if pos_flex_y else None,
        })
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('reliability_score', ascending=False)
    
    return {
        'scenario': {'interval': interval, 'defense': mapped_coverage},
        'data': df_results.head(5).to_dict('records')
    }

# ========================================
# MAIN APP
# ========================================

st.title("🏈 NFL Red Zone Analytics Dashboard")
st.markdown("*Defender Distance & Separation Strategy Analysis*")
st.markdown("---")

# Load data
with st.spinner("Loading NFL data (this may take a moment)..."):
    supp_df, input_df, output_df = load_all_data()
    
    if supp_df.empty or input_df.empty or output_df.empty:
        st.stop()
    
    redzone_df, play_summary, successful_plays = process_data(supp_df, input_df, output_df)

# Display stats
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("📊 Total Plays", len(play_summary))
with col2:
    st.metric("🏈 Touchdown Plays", len(successful_plays))
with col3:
    td_rate = (len(successful_plays) / len(play_summary) * 100) if len(play_summary) > 0 else 0
    st.metric("✅ TD Success Rate", f"{td_rate:.1f}%")

st.markdown("---")

# Sidebar controls
st.sidebar.header("🎛️ Analysis Parameters")
yards_out = st.sidebar.slider(
    "Distance from Endzone (yards)",
    min_value=5,
    max_value=20,
    value=10,
    step=1,
    help="Select the yard line distance from the endzone"
)

available_covs = sorted(play_summary['team_coverage_man_zone'].dropna().unique().tolist())
defense_type = st.sidebar.selectbox(
    "Defense Coverage Type",
    available_covs,
    help="Select the defensive coverage scheme"
)

# Analyze button
if st.sidebar.button("🔍 Analyze Play Patterns", use_container_width=True):
    st.session_state.analyze = True
else:
    st.session_state.analyze = False

# Display recommendations
if st.session_state.get('analyze'):
    with st.spinner("🔄 Analyzing play patterns..."):
        recommendations = get_enhanced_recommendations_final(
            yards_out, defense_type, play_summary, successful_plays, redzone_df
        )
    
    if 'error' not in recommendations:
        st.success(f"✅ Analysis Complete: {recommendations['scenario']['interval']} yards vs {recommendations['scenario']['defense']}")
        
        # Display results table
        st.subheader("📋 Top Play Patterns")
        df_display = pd.DataFrame(recommendations['data'])
        
        # Rename columns for display
        display_cols = {
            'formation': 'Formation',
            'route': 'Route',
            'alignment': 'Alignment',
            'td_count': 'TDs',
            'attempts': 'Attempts',
            'raw_success_rate': 'Raw Rate %',
            'simulated_success': 'Simulated Success %',
            'reliability_score': 'Reliability %',
        }
        
        df_display = df_display.rename(columns=display_cols)
        st.dataframe(
            df_display[list(display_cols.values())],
            use_container_width=True,
            hide_index=True
        )
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            df_sorted = df_display.sort_values('Simulated Success %', ascending=True)
            bars = ax.barh(
                range(len(df_sorted)),
                df_sorted['Simulated Success %'],
                color='#1f77b4'
            )
            ax.set_yticks(range(len(df_sorted)))
            ax.set_yticklabels([
                f"{row['Route']} - {row['Formation']}"
                for _, row in df_sorted.iterrows()
            ])
            ax.set_xlabel('Success Rate (%)', fontsize=12)
            ax.set_title('Success Rates by Play Pattern', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            df_sorted = df_display.sort_values('Reliability %', ascending=True)
            bars = ax.barh(
                range(len(df_sorted)),
                df_sorted['Reliability %'],
                color='#2ca02c'
            )
            ax.set_yticks(range(len(df_sorted)))
            ax.set_yticklabels([
                f"{row['Route']} - {row['Formation']}"
                for _, row in df_sorted.iterrows()
            ])
            ax.set_xlabel('Reliability Score (%)', fontsize=12)
            ax.set_title('Reliability by Play Pattern', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.error(f"❌ {recommendations['error']}")

# Footer
st.markdown("---")
st.markdown("""
    ### 📌 About This Dashboard
    This dashboard analyzes NFL Red Zone play-calling strategies by examining:
    - **Defender Distance Impact**: How separation affects catch success
    - **Coverage Effects**: Man vs Zone defensive strategies
    - **Route-Formation Combinations**: Optimal play patterns for each scenario
    - **Bayesian Reliability**: Confidence scores adjusted for sample size
    
    **Data**: NFL Big Data Bowl 2026 Competition Dataset | Weeks 1-18, 2023 Season
""")
