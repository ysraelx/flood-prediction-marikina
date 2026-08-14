# scraper/station_config.py

# ── Base URLs ──────────────────────────────────────────────────────────────────
BASE_URL      = "https://pasig-marikina-tullahanffws.pagasa.dost.gov.ph"
WL_DETAIL_URL = f"{BASE_URL}/water/detail_list.do"
RF_DETAIL_URL = "https://pasig-marikina-tullahanffws.pagasa.dost.gov.ph/rainfall/detail_list.do"

# ── Scrape window ──────────────────────────────────────────────────────────────
START_DATE = "2024-07-01"
END_DATE   = "2026-06-30"

# ── All 17 water level stations ────────────────────────────────────────────────
WATER_STATIONS = {
    "angono":        {"obscd": "11105201", "name": "Angono"},
    "burgos":        {"obscd": "11102203", "name": "Burgos"},
    "fort_santiago": {"obscd": "11204202", "name": "Fort Santiago"},
    "la_mesa_dam":   {"obscd": "11302201", "name": "La Mesa Dam"},
    "mindanao":      {"obscd": "11203201", "name": "Mindanao"},
    "montalban":     {"obscd": "11102202", "name": "Montalban"},
    "nangka":        {"obscd": "11103202", "name": "Nangka"},
    "napindan_1":    {"obscd": "11201201", "name": "Napindan-1"},
    "napindan_2":    {"obscd": "11202201", "name": "Napindan-2"},
    "pandacan":      {"obscd": "11204201", "name": "Pandacan"},
    "quirino":       {"obscd": "11302202", "name": "Quirino"},
    "rodriguez":     {"obscd": "11102201", "name": "Rodriguez"},
    "san_juan":      {"obscd": "11203203", "name": "San Juan School"},
    "san_mateo_1":   {"obscd": "11103201", "name": "San Mateo-1"},
    "sto_nino":      {"obscd": "11104201", "name": "Sto Nino"},
    "tumana":        {"obscd": "11103203", "name": "Tumana Bridge"},
    "ugong":         {"obscd": "11303201", "name": "Ugong"},
}

# ── Water level output paths ───────────────────────────────────────────────────
WL_OUTPUT_PATH     = "../data/raw/water_level_all_stations.csv"
WL_CHECKPOINT_PATH = "../data/raw/water_level_checkpoint.csv"
WL_FAILED_PATH     = "wl_failed_days.csv"
WL_PROGRESS_PATH   = "wl_progress.txt"

# ── Rainfall stations ──────────────────────────────────────────────────────────
RAINFALL_STATIONS = {
    "marikina_youth_camp": {
        "obscd": "11103107",
        "name":  "Marikina (Youth Camp)",
        "covers": ["Sto Nino", "Tumana Bridge", "Angono"],
    },
    "boso_boso": {
        "obscd": "11103103",
        "name":  "Boso Boso",
        "covers": ["Rodriguez"],
    },
    "sitio_wawa": {
        "obscd": "11102101",
        "name":  "Sitio Wawa",
        "covers": ["Montalban", "Burgos"],
    },
    "mt_aries": {
        "obscd": "11103104",
        "name":  "Mt. Aries",
        "covers": ["Rodriguez", "Burgos"],
    },
    "mt_campana": {
        "obscd": "11101102",
        "name":  "Mt. Campana",
        "covers": ["Burgos", "Montalban"],
    },
    "macabud": {
        "obscd": "11102103",
        "name":  "Macabud",
        "covers": ["Rodriguez"],
    },
    "nangka": {
        "obscd": "11103106",
        "name":  "Nangka",
        "covers": ["Nangka"],
    },
    "san_mateo_2": {
        "obscd": "11103101",
        "name":  "San Mateo-2",
        "covers": ["San Mateo-1"],
    },
}

# ── Rainfall output paths ──────────────────────────────────────────────────────
RF_OUTPUT_PATH     = "../data/raw/rainfall_marikina.csv"
RF_CHECKPOINT_PATH = "../data/raw/rainfall_checkpoint.csv"
RF_FAILED_PATH     = "rf_failed_days.csv"
RF_PROGRESS_PATH   = "rf_progress.txt"