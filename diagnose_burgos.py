import pandas as pd
import numpy as np

wl = pd.read_csv('data/processed/water_level_hourly.csv',
                 parse_dates=['datetime'])
s  = wl[wl['station_name'] == 'Burgos'].sort_values('datetime')

n     = len(s)
train = s[s['datetime'] < '2025-07-01']
val   = s[(s['datetime'] >= '2025-07-01') & (s['datetime'] < '2025-09-01')]
test  = s[s['datetime'] >= '2025-09-01']

print('BURGOS DATA ANALYSIS')
print('=' * 50)
print(f'Total rows : {n:,}')
print()
print(f'Train period: {train["datetime"].min().date()} to {train["datetime"].max().date()}')
print(f'Train WL mean: {train["water_level"].mean():.3f}')
print(f'Train WL std : {train["water_level"].std():.3f}')
print(f'Train WL min : {train["water_level"].min():.3f}')
print(f'Train WL max : {train["water_level"].max():.3f}')
print()
print(f'Test period: {test["datetime"].min().date()} to {test["datetime"].max().date()}')
print(f'Test WL mean: {test["water_level"].mean():.3f}')
print(f'Test WL std : {test["water_level"].std():.3f}')
print(f'Test WL min : {test["water_level"].min():.3f}')
print(f'Test WL max : {test["water_level"].max():.3f}')
print()
print('Risk class in test set:')
from src.data.preprocessor import assign_risk_class
test = test.copy()
test['risk'] = test['water_level'].apply(
    lambda x: assign_risk_class(x, 'Burgos')
)
print(test['risk'].value_counts().to_string())
print()
print('Null values in test:', test['water_level'].isna().sum())