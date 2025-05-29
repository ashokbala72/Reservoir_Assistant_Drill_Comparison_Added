
import streamlit as st
import pandas as pd
import openai
import os
import time
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from fpdf import FPDF

# Page config
st.set_page_config(page_title="Reservoir Simulation GenAI Assistant", layout="wide")

# Load environment
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Locations & Simulators
location_options = [
    "ghawar_field", "burgan_field", "rumaila", "south_pars_north_dome", "zubair_field", "abu_dhabi_fields",
    "mumbai_high", "krishna_godavari_basin", "bohai_bay", "browse_basin", "taranaki_basin",
    "permian_basin", "eagle_ford", "bakken_formation", "niobrara", "montney_formation", "duvernay",
    "campos_basin", "santos_basin", "orinoco_belt", "neuquen_basin",
    "niger_delta", "zohr_field", "libyan_oil_fields", "namibian_offshore",
    "brent_field", "johan_sverdrup", "ekofisk",
    "western_siberia", "sakhalin_fields", "tengiz_field", "kashagan_field"
]
sim_names = ["eclipse", "cmg", "tnavigator", "opm", "mrst"]

TAB_COLORS = {
    "LOCATION SELECTION": "#f2f2f2",
    "RESERVOIR SIMULATION ASSISTANT": "#eaf4fc",
    "ECLIPSE": "#eef7f5",
    "CMG": "#fff4e6",
    "TNAVIGATOR": "#f0f4ff",
    "OPM": "#fef0f0",
    "MRST": "#f7f3ff",
    "OVERALL SUMMARY": "#f0fff4",
    "GENAI RECOMMENDATIONS": "#f4f4f4"
}

tab_titles = ["LOCATION SELECTION", "RESERVOIR SIMULATION ASSISTANT"] + [sim.upper() for sim in sim_names] + ["OVERALL SUMMARY", "GENAI RECOMMENDATIONS"]
tabs = st.tabs(tab_titles)

# Tab 0: Location selection
with tabs[0]:
    st.markdown(f"<div style='background-color:{TAB_COLORS['LOCATION SELECTION']}; padding: 20px;'>", unsafe_allow_html=True)
    st.subheader("📍 Select a Reservoir Location")
    selected_location = st.selectbox("Choose location for simulation", location_options)
    if selected_location:
        st.session_state["selected_location"] = selected_location
        st.success(f"Using data from: **{selected_location.replace('_', ' ').title()}**")
    st.markdown("</div>", unsafe_allow_html=True)

if "selected_location" not in st.session_state:
    st.warning("Please select a location in the first tab.")
    st.stop()

# Setup
st.session_state['combined_df'] = []
st.session_state['all_summaries'] = []

def style_thresholds(df):
    def highlight(val, col):
        if col == "WATER_CUT" and val > 25:
            return "background-color: #ffcccc"
        if col == "PRESSURE" and val < 3200:
            return "background-color: #ffe4b5"
        return ""
    return df.style.apply(lambda row: [highlight(row[c], c) for c in row.index], axis=1)

def get_simulation_summary(sim_label, prompt_df, location):
    prompt = f"""
    You are a petroleum reservoir engineer assistant. Based on the following simulation output data (last 5 time steps) for the location **{location.replace('_', ' ').title()}** from {sim_label}, summarize in plain English the key production insights.
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
def get_overall_summary_cached(df_all, metric, location):
    recent_rows = df_all.groupby("SIMULATOR").tail(5)
    prompt_df = recent_rows[["TIME_DAYS", "SIMULATOR", metric]].to_markdown(index=False)
    prompt = f"""
    You are an expert reservoir analyst. The data is from **{location.replace('_', ' ').title()}**.
    Review the simulation output across simulators and explain the recent behavior of '{metric.replace('_', ' ').title()}'.

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

def get_recommendations(prompt_data, location):
    prompt = f"""
    You are a reservoir simulation expert. Based on the combined outputs of multiple simulators for **{location.replace('_', ' ').title()}**, recommend specific actions to optimize oil production and reservoir management:

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

# Intro tab
with tabs[1]:
    st.markdown(f"<div style='background-color:{TAB_COLORS['RESERVOIR SIMULATION ASSISTANT']}; padding: 20px;'>", unsafe_allow_html=True)
    st.markdown(f"""
    ## 👋 Welcome to the Reservoir Simulation & Modelling Assistant

    You are analyzing real simulation data from: **{st.session_state['selected_location'].replace('_', ' ').title()}**

    ### 📊 Simulators: ECLIPSE, CMG, tNavigator, OPM, MRST
    - Data includes: OIL_RATE, PRESSURE, WATER_CUT
    - GenAI summaries & optimization recommendations
    - Export PDF report at the end
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# Simulator tabs
for i, sim in enumerate(sim_names, start=2):
    sim_upper = sim.upper()
    file_path = f"/mnt/data/reservoir_data/{st.session_state['selected_location']}/{sim}_output.csv"
    with tabs[i]:
        st.markdown(f"<div style='background-color:{TAB_COLORS[sim_upper]}; padding: 10px;'>", unsafe_allow_html=True)
        st.header(f"📊 {sim_upper} Simulation Output")

        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                df["UPDATED_AT"] = pd.to_datetime(df["UPDATED_AT"], unit='s')
                st.dataframe(style_thresholds(df.tail(10)))
                st.session_state['combined_df'].append(df.assign(SIMULATOR=sim_upper))
                summary = get_simulation_summary(sim_upper, df.tail(5).to_markdown(index=False), st.session_state["selected_location"])
                st.markdown(f"### 🤖 GenAI Summary for {sim_upper}")
                st.markdown(summary)
                st.session_state['all_summaries'].append(f"## {sim_upper}\n{summary}")
            except Exception as e:
                st.error(f"❌ Error processing {sim_upper}: {str(e)}")
        else:
            st.warning(f"⚠️ File not found for {sim_upper} at {file_path}")
        st.markdown("</div>", unsafe_allow_html=True)

# Summary tab
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
        summary = get_overall_summary_cached(df_all, metric, st.session_state['selected_location'])
        st.markdown(summary)
    else:
        st.warning("No data loaded.")
    st.markdown("</div>", unsafe_allow_html=True)

# Recommendations tab
with tabs[-1]:
    st.markdown(f"<div style='background-color:{TAB_COLORS['GENAI RECOMMENDATIONS']}; padding: 10px;'>", unsafe_allow_html=True)
    st.header("🧠 GenAI Recommendations")
    if st.session_state['combined_df']:
        df_all = pd.concat(st.session_state['combined_df'])
        prompt_data = df_all.tail(20).to_markdown(index=False)
        recommendations = get_recommendations(prompt_data, st.session_state['selected_location'])
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
        st.warning("No data available.")
    st.markdown("</div>", unsafe_allow_html=True)
