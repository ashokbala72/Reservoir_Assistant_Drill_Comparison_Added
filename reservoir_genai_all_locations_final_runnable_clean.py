
import streamlit as st
import os
import pandas as pd
import random
import time
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="🛢️ Reservoir GenAI Assistant", layout="wide")

sim_names = ["eclipse", "cmg", "tnavigator", "opm", "mrst"]
LOCATIONS = [
"ghawar_field", "mumbai_high", "permian_basin", "burgan_field", "south_pars_north_dome",
"krishna_godavari_basin", "bohai_bay", "browse_basin", "taranaki_basin", "brent_field"
]

if "selected_location" not in st.session_state:
    st.session_state.selected_location = LOCATIONS[0]

# Tabs
tab_titles = ["🏠 Overview", "📍 Location Selection"] + [sim.upper() for sim in sim_names] + ["📈 Summary", "🧠 Recommendations"]
tabs = st.tabs(tab_titles)

# Overview
with tabs[0]:
    st.title("🛢️ Reservoir Simulation & GenAI Assistant")
    st.markdown("This assistant generates simulation data per location, shows metrics, and provides GenAI-driven insights.")

# Location Tab
with tabs[1]:
    st.header("📍 Select Location")
    st.session_state.selected_location = st.selectbox("Choose a location", LOCATIONS, index=LOCATIONS.index(st.session_state.selected_location))
    if st.button("🔄 Refresh Now"):
        location = st.session_state.selected_location
        folder = f"sim_data/{location}"
        os.makedirs(folder, exist_ok=True)
        for sim in sim_names:
            df = pd.DataFrame({
            "TIME_DAYS": [0, 30, 60, 90, 120],
            "OIL_RATE": [random.randint(850, 1000) for _ in range(5)],
            "PRESSURE": [random.randint(2700, 3700) for _ in range(5)],
            "WATER_CUT": [random.randint(5, 30) for _ in range(5)],
            "UPDATED_AT": [int(time.time())] * 5
            })
            df.to_csv(f"{folder}/{sim}_output.csv", index=False)
            st.success(f"✅ Refresh completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Load data
        location = st.session_state.selected_location
        base_path = f"sim_data/{location}"

    # Simulator Tabs
    for idx, sim in enumerate(sim_names, start=2):
        with tabs[idx]:
            st.header(f"📊 {sim.upper()} Simulation Output")
            file_path = f"{base_path}/{sim}_output.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                st.dataframe(df)
                prompt = f"You are a petroleum engineer. Summarize trends in oil rate, pressure, and water cut from this {sim} simulation at {location}.\n\n{df.head().to_csv(index=False)}"
                try:
                    response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.4,
                    )
                    st.markdown("#### 💡 GenAI Suggestions")
                    st.info(response.choices[0].message.content)
                except Exception as e:
                        st.warning(f"GenAI error: {str(e)}")

                        # Summary Tab
                        with tabs[-2]:
                            st.header("📈 Summary Graphs + GenAI Summary")
                            all_dfs = []
                            for sim in sim_names:
                                path = f"{base_path}/{sim}_output.csv"
                                if os.path.exists(path):
                                    df = pd.read_csv(path)
                                    df["SIMULATOR"] = sim.upper()
                                    all_dfs.append(df)
                                    if all_dfs:
                                        combined = pd.concat(all_dfs)
                                        metric = st.selectbox("Choose metric to compare", ["OIL_RATE", "PRESSURE", "WATER_CUT"])
                                        fig, ax = plt.subplots()
                                        for sim in combined["SIMULATOR"].unique():
                                            ax.plot(combined[combined["SIMULATOR"] == sim]["TIME_DAYS"],
                                            combined[combined["SIMULATOR"] == sim][metric],
                                            label=sim)
                                            ax.set_title(f"{metric} over Time")
                                            ax.set_xlabel("Time (days)")
                                            ax.set_ylabel(metric)
                                            ax.legend()
                                            st.pyplot(fig)




                                            try:
                                                combined_sample = combined[["TIME_DAYS", "SIMULATOR", metric]].head(20)
                                                csv_data = combined_sample.to_csv(index=False)
                                                
                                                
                                                summary_response = client.chat.completions.create(
                                                model="gpt-3.5-turbo",
                                                messages=[{"role": "user", "content": prompt_text}],
                                                temperature=0.4,
                                                )
                                                
                                                st.markdown("#### 🧠 GenAI Summary")
                                                st.info(summary_response.choices[0].message.content)
                                            except Exception as e:
                                                st.warning(f'GenAI error: {str(e)}')
                                            st.warning(f"GenAI error: {str(e)}")


                                                st.warning(f"GenAI error: {str(e)}")

                                            summary_response = client.chat.completions.create(
                                            model="gpt-3.5-turbo",
                                            messages=[{"role": "user", "content": summary_prompt}],
                                            temperature=0.4,
                                            )
                                            st.markdown("#### 🧠 GenAI Summary")
                                            st.info(summary_response.choices[0].message.content)
                                                st.warning(f"GenAI error: {str(e)}")

                                                # Recommendations Tab
                                                with tabs[-1]:
                                                    st.header("🧠 GenAI Recommendations")
                                                    try:
                                                        prompt = f"Based on simulated reservoir behavior at {location}, suggest 3 key reservoir management actions and explain briefly."
                                                        response = client.chat.completions.create(
                                                        model="gpt-3.5-turbo",
                                                        messages=[{"role": "user", "content": prompt}],
                                                        temperature=0.4,
                                                        )
                                                        st.success(response.choices[0].message.content)
                                                    except Exception as e:
                                                            st.warning(f"GenAI error: {str(e)}")