# filter_marikina.py
# Filters the full 17-station dataset down to 8 Marikina River stations

import pandas as pd

# ── Load full dataset ──────────────────────────────────────────────────────────
print("Loading full dataset...")
df = pd.read_csv('data/raw/water_level_all_stations.csv')
print(f"Full dataset shape : {df.shape}")
print(f"All stations found : {list(df['station_name'].unique())}")

# ── Define Marikina River stations ────────────────────────────────────────────
marikina_stations = [
    'Angono',
    'Burgos',
    'Montalban',
    'Nangka',
    'Rodriguez',
    'San Mateo-1',
    'Sto Nino',
    'Tumana Bridge',
]

# ── Filter ─────────────────────────────────────────────────────────────────────
print("\nFiltering to Marikina River stations...")
df_marikina = df[df['station_name'].isin(marikina_stations)].copy()

# ── Sort chronologically ───────────────────────────────────────────────────────
df_marikina['datetime'] = pd.to_datetime(df_marikina['datetime'])
df_marikina = df_marikina.sort_values(['station_name', 'datetime'])
df_marikina['datetime'] = df_marikina['datetime'].dt.strftime('%Y-%m-%d %H:%M')

# ── Validate ───────────────────────────────────────────────────────────────────
print(f"\nMarikina dataset shape : {df_marikina.shape}")
print(f"Stations kept          : {list(df_marikina['station_name'].unique())}")
print(f"Date range             : {df_marikina['datetime'].min()} "
      f"to {df_marikina['datetime'].max()}")
print(f"Null water levels      : {df_marikina['water_level'].isna().sum()}")

print("\nRecords per station:")
print(df_marikina['station_name'].value_counts().sort_index().to_string())

print("\nWater level stats per station:")
print(df_marikina.groupby('station_name')['water_level']
      .agg(['min', 'max', 'mean'])
      .round(2).to_string())

# ── Check stations not found ───────────────────────────────────────────────────
found    = list(df_marikina['station_name'].unique())
missing  = [s for s in marikina_stations if s not in found]
if missing:
    print(f"\nWARNING - These stations were not found in data: {missing}")
    print("Check spelling against actual station names above")
else:
    print("\nAll 8 Marikina stations found and included.")

# ── Save ───────────────────────────────────────────────────────────────────────
output_path = 'data/raw/water_level_marikina.csv'
df_marikina.to_csv(output_path, index=False)
print(f"\nSaved to : {output_path}")
print(f"Shape    : {df_marikina.shape}")
print("\nDone.")