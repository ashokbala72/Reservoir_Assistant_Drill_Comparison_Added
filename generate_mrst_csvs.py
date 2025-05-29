import os
import pandas as pd

base_path = "./reservoir_data"
locations = ["ghawar_field", "north_sea", "mumbai_high", "permian_basin", "kuwait_burgan"]
days = list(range(0, 201, 10))

def generate_mrst_data(location):
    return pd.DataFrame({
        "TIME_DAYS": days,
        "OIL_RATE": [5000 - i * 20 + (hash(location) % 100) for i in range(len(days))],
        "PRESSURE": [4000 - i * 10 + (hash(location) % 50) for i in range(len(days))],
        "WATER_CUT": [round(0.05 + 0.005 * i + ((hash(location) % 10) / 100), 3) for i in range(len(days))]
    })

for loc in locations:
    loc_path = os.path.join(base_path, loc)
    os.makedirs(loc_path, exist_ok=True)
    df = generate_mrst_data(loc)
    df.to_csv(os.path.join(loc_path, "mrst_output.csv"), index=False)
