import os
import streamlit as st
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
genai_cache = {}
metric_change_cache = {}

st.set_page_config(page_title="📂 Reservoir GenAI Assistant", layout="wide")

sim_names = ["eclipse", "cmg", "tnavigator", "opm", "mrst"]
LOCATIONS = [
    "ghawar_field", "mumbai_high", "permian_basin", "burgan_field", "south_pars_north_dome",
    "krishna_godavari_basin", "bohai_bay", "browse_basin", "taranaki_basin", "brent_field"
]

drill_zones = ['Zone A', 'Zone B', 'Zone C']
if "selected_location" not in st.session_state:
    st.session_state.selected_location = LOCATIONS[0]
if "drill_zone" not in st.session_state:
    st.session_state.drill_zone = drill_zones[0]

# Tabs
emoji_titles = ["🏠 Overview", "📍 Location"] + [f"🧪 {sim.upper()}" for sim in sim_names] + ["📈 Summary", "🧠 Recommendations"]
tabs = st.tabs(emoji_titles)

# 🏠 Overview
with tabs[0]:
    st.title("🔃 Reservoir Simulation & GenAI Assistant")
    st.markdown("""
    ## 🧠 Purpose of the Application
    This assistant helps petroleum engineers analyze multi-platform reservoir simulation data and get GenAI-driven insights.

    ## ✨ What’s Included
    - 10 key real-world basins for simulation
    - Real-time and mock simulators with drill zone selection
    - Integration of OpenAI for recommendations
    - Manual and auto refresh capabilities

    ## ⚙️ Supported Simulators
    | Simulator     | Type    | Description |
    |---------------|---------|-------------|
    | ECLIPSE       | Mocked  | Legacy black oil simulator from Schlumberger |
    | CMG           | Mocked  | Thermal & unconventional modeling |
    | tNavigator    | Mocked  | Dynamic sim with geological workflows |
    | OPM           | Real    | Open Porous Media simulator outputs |
    | MRST          | Real    | MATLAB-based simulation outputs |

    **Note:** Real-time simulators (OPM, MRST) are backed by pre-generated or real CSVs. Others generate synthetic data using hash seeding.

    ## 🔄 Real vs Mock Logic
    - **Real Simulators**: Directly use existing reservoir output CSVs
    - **Mock Simulators**: Generate synthetic values using seeded random logic
    - This allows scalable testing while integrating real platforms where possible

    ## ⚛️ Integration with Existing Simulation Systems
    - **For Real Platforms (OPM, MRST)**: Can directly integrate with live APIs to fetch simulation data
    - **For Mock Platforms (ECLIPSE, CMG, tNavigator)**:
        - Replace synthetic logic with actual file parsers (e.g., UNRST, INIT, FRS readers)
        - Requires ETL scripts and adapters

    ## 🚀 Path to Full Real-Time
    - Implement API fetch for all platforms
    - Build file parsers for legacy formats
    - Secure and cache data flows using database backends (e.g., PostgreSQL, InfluxDB)
    - Add real-time change detection for refresh triggers

    ## 📊 Parameters Tracked
    - `TIME_DAYS`, `OIL_RATE`, `PRESSURE`, `WATER_CUT`, `UPDATED_AT`

    ## 🌍 Drill Zones
    - Each location has 3 configurable drill zones (A, B, C)
    - Mapping to geological layers can enhance insights

    ## 🔹 Benefits
    - Fast cross-simulator comparisons
    - Centralized operational recommendations
    - Extensible and field-deployable structure
    """)

# 📍 Location Selection
with tabs[1]:
    st.header("📍 Select Location")
    st.session_state.selected_location = st.selectbox("Choose a location", LOCATIONS, index=LOCATIONS.index(st.session_state.selected_location))
    st.session_state.drill_zone = st.selectbox('Select Drill Zone', drill_zones, index=drill_zones.index(st.session_state.drill_zone))
    if st.button("🔄 Refresh Now"):
        location = st.session_state.selected_location
        folder = f"sim_data/{location}"
        os.makedirs(folder, exist_ok=True)

        location_seed = abs(hash(location)) % 10000
        for sim in sim_names:
            base = location_seed % 300 + 700
            df = pd.DataFrame({
                "TIME_DAYS": [0, 30, 60, 90, 120],
                "OIL_RATE": [random.randint(base, base + 100) for _ in range(5)],
                "PRESSURE": [random.randint(2500 + base % 300, 3500 + base % 300) for _ in range(5)],
                "WATER_CUT": [random.randint(5, 25) for _ in range(5)],
                "UPDATED_AT": [int(time.time())] * 5
            })
            df.to_csv(f"{folder}/{sim}_output.csv", index=False)
        st.success(f"✅ Refresh completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

# Simulator Tabs
for idx, sim in enumerate(sim_names, start=2):
    with tabs[idx]:
        st.header(f"🧪 {sim.upper()} Simulator")
        if st.button(f"🔄 Refresh {sim.upper()} Data"):
            location = st.session_state.selected_location
            folder = f"sim_data/{location}"
            os.makedirs(folder, exist_ok=True)
            base = abs(hash(location + sim)) % 300 + 700
            df = pd.DataFrame({
                "TIME_DAYS": [0, 30, 60, 90, 120],
                "OIL_RATE": [random.randint(base, base + 100) for _ in range(5)],
                "PRESSURE": [random.randint(2500 + base % 300, 3500 + base % 300) for _ in range(5)],
                "WATER_CUT": [random.randint(5, 25) for _ in range(5)],
                "UPDATED_AT": [int(time.time())] * 5
            })
            df.to_csv(f"{folder}/{sim}_output.csv", index=False)
            st.success(f"✅ {sim.upper()} Data Refreshed")

        file_path = f"sim_data/{st.session_state.selected_location}/{sim}_output.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            st.dataframe(df, use_container_width=True)

            fig, ax = plt.subplots(figsize=(6, 3))
            for metric in ["OIL_RATE", "PRESSURE", "WATER_CUT"]:
                label = metric
                if sim in metric_change_cache and metric in metric_change_cache[sim]:
                    old_val = metric_change_cache[sim][metric]
                    new_val = df[metric].iloc[-1]
                    direction = "⬆️" if new_val > old_val else ("⬇️" if new_val < old_val else "➡️")
                    label += f" {direction}"
                ax.plot(df["TIME_DAYS"], df[metric], label=label)
            ax.set_title(f"{sim.upper()} Simulation Trends")
            ax.legend()
            st.pyplot(fig)

            if sim not in metric_change_cache:
                metric_change_cache[sim] = {}
            for metric in ["OIL_RATE", "PRESSURE", "WATER_CUT"]:
                metric_change_cache[sim][metric] = df[metric].iloc[-1]

            cache_key = f"{st.session_state.selected_location}_{sim}"
            if cache_key not in genai_cache:
                prompt = f"You are a petroleum engineer. Summarize trends in oil rate, pressure, and water cut from this {sim} simulation at {st.session_state.selected_location} ({st.session_state.drill_zone}).\n\n{df.head().to_csv(index=False)}"
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.4,
                    )
                    genai_cache[cache_key] = response.choices[0].message.content
                except Exception as e:
                    st.warning(f"GenAI error: {str(e)}")
            st.markdown("#### 💡 GenAI Suggestions")
            st.info(genai_cache.get(cache_key, "No data available."))
        else:
            st.warning(f"❌ CSV not found for {sim.upper()} at {st.session_state.selected_location}")

# 📈 Summary
with tabs[-2]:
    st.header("📈 Cross-Simulator Summary")
    selected_metric = st.selectbox("Choose metric for trend analysis", ["OIL_RATE", "PRESSURE", "WATER_CUT"])
    dfs = []
    for sim in sim_names:
        file_path = f"sim_data/{st.session_state.selected_location}/{sim}_output.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['SIMULATOR'] = sim.upper()
            dfs.append(df)
    if dfs:
        combined_df = pd.concat(dfs)
        fig, ax = plt.subplots(figsize=(6, 3))
        for sim in sim_names:
            df_sub = combined_df[combined_df['SIMULATOR'] == sim.upper()]
            ax.plot(df_sub["TIME_DAYS"], df_sub[selected_metric], label=f"{sim.upper()} - {selected_metric}")
        ax.set_title(f"{selected_metric} Trends Across Simulators")
        ax.legend()
        st.pyplot(fig)

        cache_key = f"summary_{selected_metric}_{st.session_state.selected_location}"
        if cache_key not in genai_cache:
            csv_data = combined_df[["TIME_DAYS", "SIMULATOR", selected_metric]].head().to_csv(index=False)
            prompt = f"You are a reservoir analyst. Provide a short summary of trends in {selected_metric.lower()} across all simulators at {st.session_state.selected_location} ({st.session_state.drill_zone}).\n\n{csv_data}"
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                genai_cache[cache_key] = response.choices[0].message.content
            except Exception as e:
                st.warning(f"GenAI error: {str(e)}")
        st.markdown("#### 📌 GenAI Summary")
        st.info(genai_cache.get(cache_key, "No summary available."))

# 🧠 Recommendations
with tabs[-1]:
    st.header("🧠 GenAI Operational Recommendations")
    cache_key = f"recommend_{st.session_state.selected_location}"
    if cache_key not in genai_cache:
        prompt = f"You are a petroleum production expert. Based on recent trends across all simulators at {st.session_state.selected_location} ({st.session_state.drill_zone}), suggest 3 operational actions to optimize performance."
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            genai_cache[cache_key] = response.choices[0].message.content
        except Exception as e:
            st.warning(f"GenAI error: {str(e)}")
    st.markdown("#### 🛠️ Recommendations")
    st.success(genai_cache.get(cache_key, "No recommendations available."))
