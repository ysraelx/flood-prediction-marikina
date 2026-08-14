# scraper/retry_failed.py

import pandas as pd
import time
import logging
from datetime import datetime
from water_level_scraper import fetch_one_day, _save_checkpoint, RAW_OUTPUT_PATH

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

def retry():
    try:
        failed = pd.read_csv("scraper/failed_days.csv")
    except FileNotFoundError:
        log.info("No failed_days.csv found — nothing to retry.")
        return

    # Load existing records
    try:
        existing = pd.read_csv(RAW_OUTPUT_PATH)
        all_records = existing.to_dict("records")
    except FileNotFoundError:
        all_records = []

    recovered = []
    still_failed = []

    for _, row in failed.iterrows():
        date = datetime.strptime(row["date"], "%Y-%m-%d")
        log.info(f"Retrying: {row['obscd']} | {date.date()}")
        records = fetch_one_day(row["obscd"], date)

        if records:
            all_records.extend(records)
            recovered.append(row["date"])
        else:
            still_failed.append(row.to_dict())

        time.sleep(2)

    _save_checkpoint(all_records)
    log.info(f"Recovered: {len(recovered)} | Still failed: {len(still_failed)}")

    if still_failed:
        pd.DataFrame(still_failed).to_csv(
            "scraper/failed_days.csv", index=False
        )

if __name__ == "__main__":
    retry()