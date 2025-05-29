import os
import pandas as pd
import random
import time

locations = ['ghawar_field', 'mumbai_high', 'permian_basin', 'burgan_field', 'south_pars_north_dome', 'krishna_godavari_basin', 'bohai_bay', 'browse_basin', 'taranaki_basin', 'brent_field', 'cantarell_field', 'kuwait_oil_field', 'al_shaheen', 'niger_delta', 'offshore_sarawak', 'east_china_sea', 'jean_d_arc_basin', 'campos_basin', 'djeno_field', 'block_17_angola', 'santos_basin', 'surat_basin', 'timor_sea', 'north_falklands_basin', 'laccadive_sea', 'songliao_basin', 'tarim_basin', 'williston_basin', 'north_slope_alaska', 'volga_urals_basin', 'neftegaz_basin']
simulators = ['eclipse', 'cmg', 'tnavigator', 'opm', 'mrst']

for location in locations:
    for sim in simulators:
        days = list(range(1, 31))
        data = {
            'TIME_DAYS': days,
            'OIL_RATE': [random.randint(800, 1200) for _ in days],
            'PRESSURE': [random.randint(2800, 3200) for _ in days],
            'WATER_CUT': [round(random.uniform(0.1, 0.3), 2) for _ in days],
            'UPDATED_AT': [time.strftime('%Y-%m-%d %H:%M:%S')] * len(days)
        }
        df = pd.DataFrame(data)
        folder = f'sim_data/{location}'
        os.makedirs(folder, exist_ok=True)
        df.to_csv(f'{folder}/{sim}_output.csv', index=False)
        print(f'✅ Created: {folder}/{sim}_output.csv')
