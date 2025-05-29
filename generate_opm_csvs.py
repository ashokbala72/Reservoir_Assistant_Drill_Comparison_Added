import os
import pandas as pd

base_path = "./reservoir_data"
locations = ["ghawar_field", "north_sea", "mumbai_high", "permian_basin", "kuwait_burgan"]
days = list(range(0, 201, 10))

def generate_opm_data(location):
    return pd.DataFrame({
        "TIME_DAYS": days,
        "OIL_RATE": [6000 - i * 25 + (hash(location) % 80) for i in range(len(days))],
        "PRESSURE": [4200 - i * 12 + (hash(location) % 60) for i in range(len(days))],
        "WATER_CUT": [round(0.06 + 0.004 * i + ((hash(location) % 7) / 100), 3) for i in range(len(days))]
    })

for loc in locations:
    loc_path = os.path.join(base_path, loc)
    os.makedirs(loc_path, exist_ok=True)
    df = generate_opm_data(loc)
    df.to_csv(os.path.join(loc_path, "opm_output.csv"), index=False)
