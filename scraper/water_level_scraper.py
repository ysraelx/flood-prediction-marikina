# scraper/water_level_scraper.py

import requests
import pandas as pd
import time
import logging
import sys
import os
from datetime import datetime, timedelta
from station_config import (
    WL_DETAIL_URL,
    WATER_STATIONS,
    START_DATE,
    END_DATE,
    WL_OUTPUT_PATH,
    WL_CHECKPOINT_PATH,
    WL_FAILED_PATH,
    WL_PROGRESS_PATH,
)

sys.stdout.reconfigure(encoding="utf-8")

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("wl_scrape_log.txt", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────────────

def build_ymdhm(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M")


def clean_wl(raw) -> float:
    if raw is None:
        return None
    cleaned = str(raw).replace("(*)", "").strip()
    if cleaned in ("-", "", "null", "None"):
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def load_progress() -> set:
    """
    Load already-completed (obscd, date) pairs from progress file.
    This allows the scraper to resume from where it stopped.
    """
    completed = set()
    if os.path.exists(WL_PROGRESS_PATH):
        with open(WL_PROGRESS_PATH, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    completed.add(line)
        log.info(f"Resuming — {len(completed)} day/station pairs already done")
    return completed


def mark_progress(obscd: str, date: datetime):
    """Append completed (obscd, date) to progress file."""
    with open(WL_PROGRESS_PATH, "a") as f:
        f.write(f"{obscd}_{date.strftime('%Y-%m-%d')}\n")


def fetch_one_day(obscd: str, date: datetime) -> list:
    """
    Fetch all 10-minute water level readings for one station, one date.
    Queries at 23:50 to capture the full day (00:00 to 23:50).
    Filters response to keep only records matching the target date.
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
            WL_DETAIL_URL,
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
                "water_level":  clean_wl(item.get("wl")),
                "wl_change":    item.get("wlchange"),
                "alertwl":      clean_wl(item.get("alertwl")),
                "alarmwl":      clean_wl(item.get("alarmwl")),
                "criticalwl":   clean_wl(item.get("criticalwl")),
                "raw_wl":       item.get("wl"),
            })

        log.info(f"  OK {obscd} | {date.date()} | {len(records)} records")
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
    """Save all collected records to checkpoint CSV."""
    if not records:
        return
    df = pd.DataFrame(records)
    df.to_csv(WL_CHECKPOINT_PATH, index=False)


def finalize(records: list):
    """
    Sort by station and datetime ascending, remove duplicates,
    save to final output CSV.
    """
    if not records:
        log.warning("No records to finalize.")
        return

    df = pd.DataFrame(records)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.sort_values(['station_name', 'datetime'])
    df = df.drop_duplicates(subset=['obscd', 'datetime'])
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M')
    df.to_csv(WL_OUTPUT_PATH, index=False)
    log.info(f"Final file saved: {WL_OUTPUT_PATH}")
    log.info(f"Total records   : {len(df):,}")


# ── Main scrape loop ───────────────────────────────────────────────────────────

def scrape_all():
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end   = datetime.strptime(END_DATE,   "%Y-%m-%d")

    total_days     = (end - start).days + 1
    total_stations = len(WATER_STATIONS)
    total_requests = total_days * total_stations

    log.info("=" * 60)
    log.info("WATER LEVEL SCRAPE — ALL STATIONS")
    log.info(f"Window    : {START_DATE} to {END_DATE}")
    log.info(f"Stations  : {total_stations}")
    log.info(f"Days      : {total_days}")
    log.info(f"Requests  : ~{total_requests:,}")
    log.info(f"Est. time : ~{total_requests * 1.5 / 3600:.1f} hours")
    log.info("=" * 60)

    # Load progress — resume if interrupted
    completed   = load_progress()
    all_records = []

    # Load existing checkpoint data if resuming
    if os.path.exists(WL_CHECKPOINT_PATH):
        existing = pd.read_csv(WL_CHECKPOINT_PATH)
        all_records = existing.to_dict("records")
        log.info(f"Loaded {len(all_records):,} records from checkpoint")

    failed_days   = []
    request_count = 0
    skipped_count = 0

    current_date = start
    while current_date <= end:

        for station_key, station_info in WATER_STATIONS.items():
            obscd = station_info["obscd"]
            name  = station_info["name"]

            # Skip if already done (resume logic)
            progress_key = f"{obscd}_{current_date.strftime('%Y-%m-%d')}"
            if progress_key in completed:
                skipped_count += 1
                continue

            log.info(f"Fetching: {name} | {current_date.date()}")
            records = fetch_one_day(obscd, current_date)

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
                # Still mark as attempted so we don't retry endlessly
                # Remove this line if you want auto-retry on resume
                mark_progress(obscd, current_date)

            request_count += 1

            # Checkpoint every 50 requests
            if request_count % 50 == 0:
                save_checkpoint(all_records)
                done_pct = (
                    (request_count + skipped_count) / total_requests * 100
                )
                log.info(
                    f"  [Checkpoint] {request_count} new requests | "
                    f"{done_pct:.1f}% overall | "
                    f"{len(all_records):,} records total"
                )

            time.sleep(1.5)

        current_date += timedelta(days=1)

    # Final save
    finalize(all_records)

    # Save failed days
    if failed_days:
        pd.DataFrame(failed_days).to_csv(WL_FAILED_PATH, index=False)
        log.warning(f"Failed days saved to {WL_FAILED_PATH}")

    log.info("=" * 60)
    log.info("SCRAPE COMPLETE")
    log.info(f"New requests : {request_count:,}")
    log.info(f"Skipped      : {skipped_count:,} (already done)")
    log.info(f"Failed days  : {len(failed_days):,}")
    log.info(f"Output       : {WL_OUTPUT_PATH}")
    log.info("=" * 60)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scrape_all()