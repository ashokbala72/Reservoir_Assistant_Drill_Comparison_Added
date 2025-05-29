
import os
import pandas as pd
import random
import time

locations = [
    "ghawar_field", "burgan_field", "rumaila", "south_pars_north_dome", "zubair_field", "abu_dhabi_fields",
    "mumbai_high", "krishna_godavari_basin", "bohai_bay", "browse_basin", "taranaki_basin",
    "permian_basin", "eagle_ford", "bakken_formation", "niobrara", "montney_formation", "duvernay",
    "campos_basin", "santos_basin", "orinoco_belt", "neuquen_basin",
    "niger_delta", "zohr_field", "libyan_oil_fields", "namibian_offshore",
    "brent_field", "johan_sverdrup", "ekofisk",
    "western_siberia", "sakhalin_fields", "tengiz_field", "kashagan_field"
]

simulators = ["eclipse", "cmg", "tnavigator", "opm", "mrst"]

base_path = "reservoir_data"
os.makedirs(base_path, exist_ok=True)

for loc in locations:
    loc_path = os.path.join(base_path, loc)
    os.makedirs(loc_path, exist_ok=True)

    for sim in simulators:
        file_path = os.path.join(loc_path, f"{sim}_output.csv")
        df = pd.DataFrame({
            "TIME_DAYS": [0, 30, 60, 90, 120],
            "OIL_RATE": [round(random.uniform(850, 1000) - i * random.uniform(3, 6), 2) for i in range(5)],
            "PRESSURE": [round(random.uniform(3800, 4200) - i * random.uniform(10, 20), 2) for i in range(5)],
            "WATER_CUT": [round(random.uniform(5, 15) + i * random.uniform(1, 2.5), 2) for i in range(5)],
            "UPDATED_AT": [int(time.time())] * 5
        })
        df.to_csv(file_path, index=False)

print("✅ Local simulation files generated under ./reservoir_data/")
