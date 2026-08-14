# scraper/rainfall_scraper.py

import requests
import pandas as pd
import time
import logging
import sys
import os
from datetime import datetime, timedelta
from station_config import (
    RF_DETAIL_URL,
    RAINFALL_STATIONS,
    START_DATE,
    END_DATE,
    RF_OUTPUT_PATH,
    RF_CHECKPOINT_PATH,
    RF_FAILED_PATH,
    RF_PROGRESS_PATH,
)

sys.stdout.reconfigure(encoding="utf-8")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("rf_scrape_log.txt", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_ymdhm(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M")


def clean_rf(raw) -> float:
    if raw is None:
        return None
    cleaned = str(raw).strip()
    if cleaned in ("-", "", "null", "None"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def load_progress() -> set:
    completed = set()
    if os.path.exists(RF_PROGRESS_PATH):
        with open(RF_PROGRESS_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    completed.add(line)
        log.info(f"Resuming - {len(completed)} day/station pairs already done")
    return completed


def mark_progress(obscd: str, date: datetime):
    with open(RF_PROGRESS_PATH, "a") as f:
        f.write(f"{obscd}_{date.strftime('%Y-%m-%d')}\n")


def fetch_rainfall_one_day(obscd: str, date: datetime) -> list:
    """
    Fetch all 10-minute rainfall readings for one station on one date.
    Queries at 23:50 to capture the full day.
    Filters to keep only records matching the target date since
    the endpoint returns a rolling 24-hour window.
    """
    query_dt        = date.replace(hour=23, minute=50)
    ymdhm_str       = build_ymdhm(query_dt)
    target_date_str = date.strftime("%Y-%m-%d")

    payload = {
        "obscd": obscd,
        "ymdhm": ymdhm_str,
    }

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Referer":      "https://pasig-marikina-tullahanffws.pagasa.dost.gov.ph/",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
    }

    try:
        response = requests.post(
            RF_DETAIL_URL,
            data=payload,
            headers=headers,
            timeout=30,
        )
        response.raise_for_status()
        raw_data = response.json()

        if not isinstance(raw_data, list) or len(raw_data) == 0:
            log.warning(f"Empty: obscd={obscd} date={date.date()}")
            return []

        records = []
        for item in raw_data:
            timestr = item.get("timestr", "")
            if not timestr:
                continue
            if not timestr.startswith(target_date_str):
                continue

            records.append({
                "station_name": item.get("obsnm"),
                "obscd":        obscd,
                "datetime":     timestr,
                "rainfall_mm":  clean_rf(item.get("rf")),
                "rfday":        clean_rf(item.get("rfday")),
                "raw_rf":       item.get("rf"),
            })

        log.info(
            f"  OK {obscd} | {date.date()} | {len(records)} records"
        )
        return records

    except requests.exceptions.Timeout:
        log.error(f"  TIMEOUT: obscd={obscd} date={date.date()}")
        return []
    except requests.exceptions.RequestException as e:
        log.error(f"  ERROR: obscd={obscd} date={date.date()} | {e}")
        return []
    except Exception as e:
        log.error(f"  UNEXPECTED: obscd={obscd} date={date.date()} | {e}")
        return []


def save_checkpoint(records: list):
    if not records:
        return
    df = pd.DataFrame(records)
    df.to_csv(RF_CHECKPOINT_PATH, index=False)


def finalize(records: list):
    if not records:
        log.warning("No records to finalize.")
        return

    df = pd.DataFrame(records)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['station_name', 'datetime'])
    df = df.drop_duplicates(subset=['obscd', 'datetime'])
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M')
    df.to_csv(RF_OUTPUT_PATH, index=False)
    log.info(f"Final file saved : {RF_OUTPUT_PATH}")
    log.info(f"Total records    : {len(df):,}")


# ── Main scrape loop ───────────────────────────────────────────────────────────

def scrape_rainfall():
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end   = datetime.strptime(END_DATE,   "%Y-%m-%d")

    total_days     = (end - start).days + 1
    total_stations = len(RAINFALL_STATIONS)
    total_requests = total_days * total_stations

    log.info("=" * 60)
    log.info("RAINFALL SCRAPE — MARIKINA BASIN STATIONS")
    log.info(f"Window    : {START_DATE} to {END_DATE}")
    log.info(f"Stations  : {total_stations}")
    log.info(f"Days      : {total_days}")
    log.info(f"Requests  : ~{total_requests:,}")
    log.info(f"Est. time : ~{total_requests * 1.5 / 3600:.1f} hours")
    log.info("=" * 60)

    completed   = load_progress()
    all_records = []

    if os.path.exists(RF_CHECKPOINT_PATH):
        existing    = pd.read_csv(RF_CHECKPOINT_PATH)
        all_records = existing.to_dict("records")
        log.info(f"Loaded {len(all_records):,} records from checkpoint")

    failed_days   = []
    request_count = 0
    skipped_count = 0

    current_date = start
    while current_date <= end:

        for station_key, station_info in RAINFALL_STATIONS.items():
            obscd = station_info["obscd"]
            name  = station_info["name"]

            progress_key = f"{obscd}_{current_date.strftime('%Y-%m-%d')}"
            if progress_key in completed:
                skipped_count += 1
                continue

            log.info(f"Fetching: {name} | {current_date.date()}")
            records = fetch_rainfall_one_day(obscd, current_date)

            if records:
                all_records.extend(records)
                mark_progress(obscd, current_date)
            else:
                failed_days.append({
                    "station_key": station_key,
                    "obscd":       obscd,
                    "name":        name,
                    "date":        str(current_date.date()),
                })
                mark_progress(obscd, current_date)

            request_count += 1

            if request_count % 50 == 0:
                save_checkpoint(all_records)
                done_pct = (
                    (request_count + skipped_count) / total_requests * 100
                )
                log.info(
                    f"  [Checkpoint] {request_count} new | "
                    f"{done_pct:.1f}% overall | "
                    f"{len(all_records):,} records"
                )

            time.sleep(1.5)

        current_date += timedelta(days=1)

    finalize(all_records)

    if failed_days:
        pd.DataFrame(failed_days).to_csv(RF_FAILED_PATH, index=False)
        log.warning(f"Failed days saved to {RF_FAILED_PATH}")

    log.info("=" * 60)
    log.info("RAINFALL SCRAPE COMPLETE")
    log.info(f"New requests : {request_count:,}")
    log.info(f"Skipped      : {skipped_count:,}")
    log.info(f"Failed days  : {len(failed_days):,}")
    log.info(f"Output       : {RF_OUTPUT_PATH}")
    log.info("=" * 60)


if __name__ == "__main__":
    scrape_rainfall()