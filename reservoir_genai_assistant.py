import streamlit as st
import pandas as pd
import openai
import os
import time
import hashlib
import random
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from fpdf import FPDF

# Page config
st.set_page_config(page_title="Reservoir Simulation GenAI Assistant", layout="wide")

# Load environment
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# UI title and refresh
st.markdown("<h1 style='text-align: center;'>🛢️ Reservoir Simulation & Modelling Assistant</h1>", unsafe_allow_html=True)
refresh_now = st.button("🔄 **Refresh Now**")

# Session reset
if refresh_now or "combined_df" not in st.session_state:
    st.session_state['combined_df'] = []
    st.session_state['all_summaries'] = []

# Simulators
sim_names = ["eclipse", "cmg", "tnavigator", "opm", "mrst"]
TAB_COLORS = {
    "RESERVOIR SIMULATION ASSISTANT": "#eaf4fc",
    "ECLIPSE": "#eef7f5",
    "CMG": "#fff4e6",
    "TNAVIGATOR": "#f0f4ff",
    "OPM": "#fef0f0",
    "MRST": "#f7f3ff",
    "OVERALL SUMMARY": "#f0fff4",
    "GENAI RECOMMENDATIONS": "#f4f4f4"
}
tab_titles = ["RESERVOIR SIMULATION ASSISTANT"] + [sim.upper() for sim in sim_names] + ["OVERALL SUMMARY", "GENAI RECOMMENDATIONS"]

# Simulate new data
os.makedirs("sim_data", exist_ok=True)
if refresh_now:
    timestamp = int(time.time())
    for sim in sim_names:
        path = f"sim_data/{sim}_output.csv"
        df = pd.DataFrame({
            "TIME_DAYS": [0, 30, 60, 90, 120],
            "OIL_RATE": [random.randint(800, 1000) for _ in range(5)],
            "PRESSURE": [random.randint(2800, 4000) for _ in range(5)],
            "WATER_CUT": [random.randint(0, 35) for _ in range(5)],
            "UPDATED_AT": [timestamp] * 5
        })
        df.to_csv(path, index=False)

# Helpers
def style_thresholds(df):
    def highlight(val, col):
        if col == "WATER_CUT" and val > 25:
            return "background-color: #ffcccc"
        if col == "PRESSURE" and val < 3200:
            return "background-color: #ffe4b5"
        return ""
    return df.style.apply(lambda row: [highlight(row[c], c) for c in row.index], axis=1)

def get_simulation_summary(sim_label, prompt_df):
    prompt = f"""
    You are a petroleum reservoir engineer assistant. Based on the following simulation output data (last 5 time steps) from {sim_label}, summarize in plain English the key production insights.
    Highlight changes in oil rate, pressure, water cut, and any anomaly.

    {prompt_df}

    Provide bullet points.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ GenAI summary failed for {sim_label}: {str(e)}"

@st.cache_data(show_spinner=True)
def get_overall_summary_cached(df_all, metric):
    recent_rows = df_all.groupby("SIMULATOR").tail(5)
    prompt_df = recent_rows[["TIME_DAYS", "SIMULATOR", metric]].to_markdown(index=False)
    prompt = f"""
    You are an expert reservoir analyst. Review the simulation output across simulators and explain the recent behavior of '{metric.replace('_', ' ').title()}'.

    {prompt_df}

    Explain any rise/fall, differences between simulators, and what it could mean in simple terms.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ GenAI summary failed: {str(e)}"

def get_recommendations(prompt_data):
    prompt = f"""
    You are a reservoir simulation expert. Based on the combined outputs of multiple simulators, recommend specific actions to optimize oil production and reservoir management:

    {prompt_data}

    Highlight optimization strategies, early warning signs, and reservoir performance improvements.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ GenAI recommendation failed: {str(e)}"

# Tabs
tabs = st.tabs(tab_titles)

# Overview
with tabs[0]:
    st.markdown(f"<div style='background-color:{TAB_COLORS['RESERVOIR SIMULATION ASSISTANT']}; padding: 20px;'>", unsafe_allow_html=True)
    st.markdown("""
    ## 👋 Welcome to the Reservoir Simulation & Modelling Assistant

    This assistant helps engineers understand what's happening inside a reservoir — even if you're not a simulation expert.

    ### 🔍 What Does It Do?
    - Reads simulated data from:
      - ECLIPSE, CMG, tNavigator, OPM, MRST
    - Warns you if any simulator is missing

    ### 📊 Data Extracted
    | Parameter     | Meaning                                          |
    |---------------|--------------------------------------------------|
    | **OIL_RATE**  | Oil produced per day (barrels/day)              |
    | **PRESSURE**  | Internal reservoir pressure                     |
    | **WATER_CUT** | % of water in production — high = inefficiency |

    ### 🤖 GenAI Insights
    - Bullet summaries for each simulator
    - Anomaly detection: high water cut, low pressure
    - Comparison graphs
    - AI-generated optimization recommendations

    ### 📄 Export
    You can export everything as a PDF report.
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Per Simulator
for i, sim in enumerate(sim_names, start=1):
    sim_upper = sim.upper()
    file_path = f"sim_data/{sim}_output.csv"
    with tabs[i]:
        st.markdown(f"<div style='background-color:{TAB_COLORS[sim_upper]}; padding: 10px;'>", unsafe_allow_html=True)
        st.header(f"📊 {sim_upper} Simulation Output")

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df["UPDATED_AT"] = pd.to_datetime(df["UPDATED_AT"], unit='s')
                st.dataframe(style_thresholds(df.tail(10)))
                st.session_state['combined_df'].append(df.assign(SIMULATOR=sim_upper))

                latest_df = df.tail(5)
                summary = get_simulation_summary(sim_upper, latest_df.to_markdown(index=False))
                st.markdown(f"### 🤖 GenAI Summary for {sim_upper}")
                st.markdown(summary)
                st.session_state['all_summaries'].append(f"## {sim_upper}\n{summary}")
            except Exception as e:
                st.error(f"❌ Error processing {sim_upper}: {str(e)}")
        else:
            st.warning(f"⚠️ CSV not found for {sim_upper}")
        st.markdown("</div>", unsafe_allow_html=True)

# Overall Summary
with tabs[-2]:
    st.markdown(f"<div style='background-color:{TAB_COLORS['OVERALL SUMMARY']}; padding: 10px;'>", unsafe_allow_html=True)
    st.header("📊 Consolidated Metric Comparison")

    if st.session_state['combined_df']:
        df_all = pd.concat(st.session_state['combined_df'])
        metric = st.selectbox("Select metric:", ["OIL_RATE", "PRESSURE", "WATER_CUT"])
        fig, ax = plt.subplots()
        for sim in df_all["SIMULATOR"].unique():
            ax.plot(df_all[df_all["SIMULATOR"] == sim]["TIME_DAYS"], df_all[df_all["SIMULATOR"] == sim][metric], label=sim)
        ax.set_title(f"{metric} over Time")
        ax.set_xlabel("Time (days)")
        ax.set_ylabel(metric)
        ax.legend()
        st.pyplot(fig)

        st.subheader("🤖 GenAI Summary of Trends")
        summary = get_overall_summary_cached(df_all, metric)
        st.markdown(summary)
    else:
        st.warning("No simulation data found.")
    st.markdown("</div>", unsafe_allow_html=True)

# GenAI Recommendations
with tabs[-1]:
    st.markdown(f"<div style='background-color:{TAB_COLORS['GENAI RECOMMENDATIONS']}; padding: 10px;'>", unsafe_allow_html=True)
    st.header("🧠 GenAI Recommendations")
    if st.session_state['combined_df']:
        df_all = pd.concat(st.session_state['combined_df'])
        prompt_data = df_all.tail(20).to_markdown(index=False)
        recommendations = get_recommendations(prompt_data)
        st.markdown(recommendations)

        if st.button("📄 Export PDF Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, "Reservoir Simulation GenAI Summary")
            for section in st.session_state['all_summaries']:
                pdf.multi_cell(0, 10, section.strip())
            pdf.multi_cell(0, 10, "\nRecommendations:\n" + recommendations.strip())
            pdf.output("genai_reservoir_summary.pdf")
            st.success("✅ PDF report saved as genai_reservoir_summary.pdf")
    else:
        st.warning("Simulation data not yet available.")
    st.markdown("</div>", unsafe_allow_html=True)
