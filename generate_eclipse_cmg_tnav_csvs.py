import os
import pandas as pd

base_path = "./reservoir_data"
locations = ["ghawar_field", "north_sea", "mumbai_high", "permian_basin", "kuwait_burgan"]
days = list(range(0, 201, 10))

def generate_data(platform, location):
    return pd.DataFrame({
        "TIME_DAYS": days,
        "OIL_RATE": [4500 - i * 15 + (hash(location + platform) % 90) for i in range(len(days))],
        "PRESSURE": [3900 - i * 8 + (hash(location + platform) % 70) for i in range(len(days))],
        "WATER_CUT": [round(0.04 + 0.006 * i + ((hash(location + platform) % 8) / 100), 3) for i in range(len(days))]
    })

platforms = ["eclipse", "cmg", "tnavigator"]

for loc in locations:
    loc_path = os.path.join(base_path, loc)
    os.makedirs(loc_path, exist_ok=True)
    for platform in platforms:
        df = generate_data(platform, loc)
        df.to_csv(os.path.join(loc_path, f"{platform}_output.csv"), index=False)
