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

# 🧠 Recommendations
with tabs[-1]:
    st.header("🧠 GenAI Operational Recommendations")
    rec_focus = st.radio("Focus of Recommendations", ["Production Optimization", "Water Management", "Pressure Maintenance"], index=0)

    # Aggregate metrics for enhanced prompt context
    latest_metrics = {"oil_rate": [], "pressure": [], "water_cut": []}
    for sim in sim_names:
        file_path = f"sim_data/{st.session_state.selected_location}/{sim}_output.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            latest_metrics["oil_rate"].append(df["OIL_RATE"].iloc[-1])
            latest_metrics["pressure"].append(df["PRESSURE"].iloc[-1])
            latest_metrics["water_cut"].append(df["WATER_CUT"].iloc[-1])

    if all(latest_metrics.values()):
        avg_oil = round(sum(latest_metrics["oil_rate"]) / len(latest_metrics["oil_rate"]), 2)
        avg_pres = round(sum(latest_metrics["pressure"]) / len(latest_metrics["pressure"]), 2)
        avg_wc = round(sum(latest_metrics["water_cut"]) / len(latest_metrics["water_cut"]), 2)
    else:
        avg_oil, avg_pres, avg_wc = "N/A", "N/A", "N/A"

    cache_key = f"recommend_{st.session_state.selected_location}_{rec_focus}"
    if st.button("🔄 Regenerate Recommendations") or cache_key not in genai_cache:
        prompt = f"""
        You are a petroleum production expert.
        Based on simulator trends at {st.session_state.selected_location} ({st.session_state.drill_zone}):
        - Average Oil Rate: {avg_oil}
        - Average Pressure: {avg_pres}
        - Average Water Cut: {avg_wc}

        Provide 3 concrete recommendations focused on {rec_focus}.
        Format:
        - ✅ Action:
        - 🔍 Why:
        - 📈 Expected Impact:
        """
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            genai_cache[cache_key] = response.choices[0].message.content
        except Exception as e:
            st.warning(f"GenAI error: {str(e)}")

    st.markdown("#### 🛠️ Targeted Recommendations")
    st.success(genai_cache.get(cache_key, "No recommendations available."))
