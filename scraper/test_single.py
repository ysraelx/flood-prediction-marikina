# scraper/test_single.py
import sys
sys.path.insert(0, '.')

from datetime import datetime
from water_level_scraper import fetch_one_day

# Test: Sto Nino, August 1 2025
records = fetch_one_day("11104201", datetime(2025, 8, 1))

if records:
    print(f"\n✓ SUCCESS — {len(records)} records returned")
    print(f"\nFirst record : {records[0]}")
    print(f"Last record  : {records[-1]}")
    print(f"\nEarliest timestamp : {records[-1]['datetime']}")
    print(f"Latest timestamp   : {records[0]['datetime']}")
else:
    print("\n✗ FAILED — no records returned, check scrape_log.txt")