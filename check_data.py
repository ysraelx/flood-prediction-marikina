import pandas as pd

wl = pd.read_csv('data/raw/water_level_marikina.csv', parse_dates=['datetime'])

# does the (*) flag correlate with water_level == 0?
wl['has_flag'] = wl['raw_wl'].astype(str).str.contains(r'\(\*\)')
wl['is_zero'] = wl['water_level'] == 0.0

print("=== Zero water_level count per station ===")
print(wl.groupby('station_name')['is_zero'].sum())

print("\n=== Flag vs zero-value crosstab ===")
print(pd.crosstab(wl['has_flag'], wl['is_zero']))

print("\n=== Flag rate per station ===")
print(wl.groupby('station_name')['has_flag'].mean())

# check if flag correlates with time (e.g. flagged only in certain months)
wl['month'] = wl['datetime'].dt.to_period('M')
print("\n=== Flag rate per month ===")
print(wl.groupby('month')['has_flag'].mean())

# how many rows have wl_change with abs value > 5 (implausible 10-min swing)
wl_change_numeric = pd.to_numeric(wl['wl_change'].replace('-', pd.NA), errors='coerce')
extreme = wl_change_numeric.abs() > 5
print("\n=== Extreme wl_change (>5m in 10min) count per station ===")
print(wl.groupby('station_name').apply(lambda g: (wl_change_numeric.loc[g.index].abs() > 5).sum()))