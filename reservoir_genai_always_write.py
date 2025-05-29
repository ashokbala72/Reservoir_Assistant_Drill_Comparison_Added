
import streamlit as st
import os
import pandas as pd
import random
import time

st.set_page_config(page_title="🔨 Auto File Generator", layout="wide")

sim_names = ["eclipse", "cmg", "tnavigator", "opm", "mrst"]
LOCATIONS = ["ghawar_field", "mumbai_high", "permian_basin"]

selected_location = st.selectbox("Choose location", LOCATIONS, index=0)
st.success(f"📍 Selected location: {selected_location}")

base_path = f"sim_data/{selected_location}"
os.makedirs(base_path, exist_ok=True)
st.success(f"📂 Created or confirmed folder: {base_path}")

timestamp = int(time.time())
for sim in sim_names:
    try:
        df = pd.DataFrame({
            "TIME_DAYS": [0, 30, 60, 90, 120],
            "OIL_RATE": [random.randint(850, 1000) for _ in range(5)],
            "PRESSURE": [random.randint(2700, 3700) for _ in range(5)],
            "WATER_CUT": [random.randint(5, 30) for _ in range(5)],
            "UPDATED_AT": [timestamp] * 5
        })
        path = f"{base_path}/{sim}_output.csv"
        df.to_csv(path, index=False)
        st.success(f"✅ Wrote: {path}")
    except Exception as e:
        st.error(f"❌ Failed to write {sim} CSV")
        st.exception(e)

# Display file previews
for sim in sim_names:
    path = f"{base_path}/{sim}_output.csv"
    st.subheader(f"📄 {sim.upper()}: {path}")
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            st.dataframe(df)
        except Exception as e:
            st.error("❌ Error reading file")
            st.exception(e)
    else:
        st.error(f"❌ File not found: {path}")
