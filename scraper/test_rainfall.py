# scraper/test_rainfall.py
import sys
sys.path.insert(0, '.')

from datetime import datetime
from rainfall_scraper import fetch_rainfall_one_day

# Test: Marikina Youth Camp, August 1 2025
records = fetch_rainfall_one_day("11103107", datetime(2025, 8, 1))

if records:
    print(f"\nSUCCESS - {len(records)} records returned")
    print(f"First : {records[0]}")
    print(f"Last  : {records[-1]}")
    print(f"Earliest timestamp : {records[-1]['datetime']}")
    print(f"Latest timestamp   : {records[0]['datetime']}")
else:
    print("\nFAILED - check rainfall_scrape_log.txt")