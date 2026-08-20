# Flood Prediction and Hotspot Mapping — Marikina River

**Augmentation of a Hybrid LSTM–Random Forest Algorithm for 
Real-Time Flood Prediction and Hotspot Mapping Along Marikina River**

Pamantasan ng Lungsod ng Muntinlupa
College of Information Technology and Computer Studies
BS Computer Science — 2026

---

## Authors
- Mananquil, Jesrael B.
- Peñaflor, Jade Jasmine M.
- Songcayauon, John Blaise R.

---

## Project Overview

A real-time web-based flood prediction and hotspot mapping system 
for communities along the Marikina River. Uses a novel augmented 
hybrid LSTM–Random Forest architecture to predict water levels 
1, 3, and 6 hours ahead and classify flood risk as 
Normal / Alert / Critical per monitoring station.

---

## Data Sources

- **Water Level:** PAGASA FFWS — 7 Marikina River stations
- **Rainfall:** PAGASA FFWS — 8 Marikina Basin stations
- **Window:** July 1, 2024 – June 30, 2026
- **Resolution:** 10-minute intervals (resampled to hourly)

### Water Level Stations
| Station | Alert (m) | Critical (m) |
|---|---|---|
| Sto Nino | 15.00 | 17.00 |
| Tumana Bridge | 17.26 | 19.26 |
| Rodriguez | 28.80 | 30.70 |
| Nangka | 16.50 | 17.70 |
| San Mateo-1 | 18.00 | 20.00 |
| Montalban | 22.40 | 23.60 |
| Burgos | 27.40 | 28.40 |

---

## Project Structure