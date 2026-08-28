import pandas as pd
import numpy as np

wl = pd.read_csv('data/processed/water_level_hourly.csv', parse_dates=['datetime'])
s  = wl[wl['station_name'] == 'Sto Nino'].sort_values('datetime')

n    = len(s)
test = s.iloc[int(n * 0.90):]

print('Test period:', test['datetime'].min(), 'to', test['datetime'].max())
print('Test WL mean:', round(test['water_level'].mean(), 3))
print('Test WL std :', round(test['water_level'].std(),  3))
print('Test WL min :', round(test['water_level'].min(),  3))
print('Test WL max :', round(test['water_level'].max(),  3))
print('Variance    :', round(test['water_level'].var(),  4))