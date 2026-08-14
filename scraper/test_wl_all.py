# scraper/test_wl_all.py
# Tests one day across ALL 17 stations before full run

import sys
sys.path.insert(0, '.')

from datetime import datetime
from water_level_scraper import fetch_one_day
from station_config import WATER_STATIONS

print("Testing all 17 stations for 2024-07-15...")
print("=" * 50)

test_date   = datetime(2024, 7, 15)
all_pass    = True
total       = 0

for key, info in WATER_STATIONS.items():
    records = fetch_one_day(info["obscd"], test_date)
    status  = "OK" if records else "EMPTY"
    count   = len(records)
    total  += count

    if not records:
        all_pass = False

    print(f"  {status:5} | {info['name']:20} | {count} records")

print("=" * 50)
print(f"Total records : {total}")
print(f"All stations  : {'PASS' if all_pass else 'SOME EMPTY - check above'}")