import streamlit as st
import pandas as pd
import os
import time
import random
import matplotlib.pyplot as plt
from fpdf import FPDF
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

st.set_page_config(page_title="Reservoir Simulation GenAI Assistant", layout="wide")

@st.cache_resource
def load_local_llm():
    model_id = "tiiuae/falcon-7b-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto"
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=256)

llm_pipe = load_local_llm()

def run_llm(prompt, temperature=0.4, max_tokens=256):
    try:
        result = llm_pipe(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=temperature)
        return result[0]['generated_text'].strip()
    except Exception as e:
        return f"⚠️ LLM Error: {e}"

def get_simulation_summary(sim_label, df_recent):
    records = df_recent.tail(5).to_dict(orient='records')
    observations = "\n".join(
        f"Day {r['TIME_DAYS']}: Oil={r['OIL_RATE']} bbl/day, Pressure={r['PRESSURE']} psi, WaterCut={r['WATER_CUT']}%"
        for r in records
    )
    prompt = f"Summarize this simulation data from {sim_label} in 2-3 short insights:\n\n{observations}"
    return run_llm(prompt)

def get_recommendations(df):
    records = df.tail(10).to_dict(orient='records')
    observations = "\n".join(
        f"{r['SIMULATOR']}, Day {r['TIME_DAYS']}: Oil={r['OIL_RATE']} bbl/day, Pressure={r['PRESSURE']} psi, WaterCut={r['WATER_CUT']}%"
        for r in records if 'SIMULATOR' in r
    )
    prompt = f"Give 2-3 production optimization suggestions:\n\n{observations}"
    return run_llm(prompt)

def style_thresholds(df):
    def highlight(val, col):
        if col == "WATER_CUT" and val > 25: return "background-color: #ffcccc"
        if col == "PRESSURE" and val < 3200: return "background-color: #ffe4b5"
        return ""
    return df.style.apply(lambda row: [highlight(row[c], c) for c in row.index], axis=1)

sim_names = ["eclipse", "cmg", "tnavigator", "opm", "mrst"]
tab_titles = ["MAIN"] + [s.upper() for s in sim_names] + ["SUMMARY", "RECOMMENDATIONS"]
tabs = st.tabs(tab_titles)

refresh_now = st.button("🔄 Refresh Now")
if refresh_now or "combined_df" not in st.session_state:
    st.session_state["combined_df"] = []
    st.session_state["all_summaries"] = []

os.makedirs("sim_data", exist_ok=True)
if refresh_now:
    now = int(time.time())
    for sim in sim_names:
        df = pd.DataFrame({
            "TIME_DAYS": [0, 30, 60, 90, 120],
            "OIL_RATE": [random.randint(800, 1000) for _ in range(5)],
            "PRESSURE": [random.randint(2800, 4000) for _ in range(5)],
            "WATER_CUT": [random.randint(0, 35) for _ in range(5)],
            "UPDATED_AT": [now] * 5
        })
        df.to_csv(f"sim_data/{sim}_output.csv", index=False)

for i, sim in enumerate(sim_names, start=1):
    with tabs[i]:
        file = f"sim_data/{sim}_output.csv"
        if os.path.exists(file):
            df = pd.read_csv(file)
            df["UPDATED_AT"] = pd.to_datetime(df["UPDATED_AT"], unit='s')
            st.dataframe(style_thresholds(df.tail(10)))
            df["SIMULATOR"] = sim.upper()
            st.session_state['combined_df'].append(df)
            summary = get_simulation_summary(sim.upper(), df)
            st.markdown("### GenAI Summary")
            st.markdown(summary)
            st.session_state['all_summaries'].append(f"## {sim.upper()}\n{summary}")
        else:
            st.warning("No data found.")

with tabs[-2]:
    st.header("Metric Comparison")
    if st.session_state['combined_df']:
        df_all = pd.concat(st.session_state['combined_df'])
        metric = st.selectbox("Select metric", ["OIL_RATE", "PRESSURE", "WATER_CUT"])
        fig, ax = plt.subplots()
        for sim in df_all["SIMULATOR"].unique():
            data = df_all[df_all["SIMULATOR"] == sim]
            ax.plot(data["TIME_DAYS"], data[metric], label=sim)
        ax.set_title(f"{metric} over Time")
        ax.legend()
        st.pyplot(fig)

with tabs[-1]:
    st.header("Recommendations")
    if st.session_state['combined_df']:
        df_all = pd.concat(st.session_state['combined_df'])
        st.markdown(get_recommendations(df_all))
        if st.button("Export PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, "Reservoir Summary")
            for section in st.session_state["all_summaries"]:
                pdf.multi_cell(0, 10, section.strip())
            pdf.output("genai_reservoir_summary.pdf")
            st.success("PDF saved as genai_reservoir_summary.pdf")
