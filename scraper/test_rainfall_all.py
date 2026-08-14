# scraper/test_rainfall_all.py
import sys
sys.path.insert(0, '.')

from datetime import datetime
from rainfall_scraper import fetch_rainfall_one_day
from station_config import RAINFALL_STATIONS

print("Testing all 8 rainfall stations for 2024-07-15...")
print("=" * 55)

test_date = datetime(2024, 7, 15)
all_pass  = True
total     = 0

for key, info in RAINFALL_STATIONS.items():
    records = fetch_rainfall_one_day(info["obscd"], test_date)
    status  = "OK" if records else "EMPTY"
    count   = len(records)
    total  += count

    if not records:
        all_pass = False

    print(f"  {status:5} | {info['name']:25} | {count} records")

print("=" * 55)
print(f"Total records : {total}")
print(f"All stations  : {'PASS' if all_pass else 'SOME EMPTY - check above'}")