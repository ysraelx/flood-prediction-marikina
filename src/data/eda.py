# src/data/eda.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import numpy as np
import os

# ── Setup ──────────────────────────────────────────────────────────────────────
os.makedirs("notebooks/eda_plots", exist_ok=True)
plt.style.use('seaborn-v0_8-darkgrid')
COLORS = {
    'Normal':   '#2ecc71',
    'Alert':    '#f39c12',
    'Critical': '#e74c3c',
}

print("Loading preprocessed data...")
wl = pd.read_csv('data/processed/water_level_hourly.csv',
                 parse_dates=['datetime'])
rf = pd.read_csv('data/processed/rainfall_hourly.csv',
                 parse_dates=['datetime'])

STATIONS = sorted(wl['station_name'].unique())
print(f"Water level stations : {STATIONS}")
print(f"Rainfall stations    : {sorted(rf['station_name'].unique())}")
print(f"WL date range        : {wl['datetime'].min()} to {wl['datetime'].max()}")
print()


# ── Plot 1 — Water level time series per station ───────────────────────────────
print("Generating Plot 1: Water level time series...")

fig, axes = plt.subplots(len(STATIONS), 1,
                          figsize=(18, 4 * len(STATIONS)),
                          sharex=True)

for ax, station in zip(axes, STATIONS):
    s = wl[wl['station_name'] == station].copy()
    s = s.sort_values('datetime')

    alertwl    = s['alertwl'].iloc[0]
    criticalwl = s['criticalwl'].iloc[0]

    ax.plot(s['datetime'], s['water_level'],
            color='steelblue', linewidth=0.6, alpha=0.8)
    ax.axhline(alertwl,    color='orange', linestyle='--',
               linewidth=1.2, label=f'Alert ({alertwl}m)')
    ax.axhline(criticalwl, color='red',    linestyle='--',
               linewidth=1.2, label=f'Critical ({criticalwl}m)')

    ax.set_ylabel('Water Level (m)', fontsize=9)
    ax.set_title(station, fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

plt.suptitle('Marikina River Water Level — Jul 2024 to Jun 2026',
             fontsize=14, fontweight='bold', y=1.002)
plt.tight_layout()
plt.savefig('notebooks/eda_plots/01_water_level_timeseries.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_water_level_timeseries.png")


# ── Plot 2 — Rainfall time series per station ──────────────────────────────────
print("Generating Plot 2: Rainfall time series...")

RF_STATIONS = sorted(rf['station_name'].unique())
fig, axes = plt.subplots(len(RF_STATIONS), 1,
                          figsize=(18, 3 * len(RF_STATIONS)),
                          sharex=True)

for ax, station in zip(axes, RF_STATIONS):
    s = rf[rf['station_name'] == station].copy()
    s = s.sort_values('datetime')

    ax.bar(s['datetime'], s['rainfall_mm'],
           width=0.04, color='steelblue', alpha=0.7)
    ax.set_ylabel('Rainfall (mm/hr)', fontsize=9)
    ax.set_title(station, fontsize=11, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

plt.suptitle('Marikina Basin Rainfall — Jul 2024 to Jun 2026',
             fontsize=14, fontweight='bold', y=1.002)
plt.tight_layout()
plt.savefig('notebooks/eda_plots/02_rainfall_timeseries.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_rainfall_timeseries.png")


# ── Plot 3 — Risk class distribution per station ───────────────────────────────
print("Generating Plot 3: Risk class distribution...")

risk_counts = (wl.groupby(['station_name', 'risk_class'])
                 .size()
                 .reset_index(name='count'))
risk_pct = risk_counts.copy()
totals   = risk_pct.groupby('station_name')['count'].transform('sum')
risk_pct['pct'] = risk_pct['count'] / totals * 100

fig, ax = plt.subplots(figsize=(14, 6))
pivot = risk_pct.pivot(index='station_name',
                        columns='risk_class',
                        values='pct').fillna(0)

for col in ['Normal', 'Alert', 'Critical']:
    if col not in pivot.columns:
        pivot[col] = 0
pivot = pivot[['Normal', 'Alert', 'Critical']]

pivot.plot(kind='bar', ax=ax, stacked=True,
           color=[COLORS['Normal'], COLORS['Alert'], COLORS['Critical']],
           edgecolor='white', linewidth=0.5)

ax.set_xlabel('')
ax.set_ylabel('Percentage (%)', fontsize=11)
ax.set_title('Flood Risk Class Distribution per Station',
             fontsize=13, fontweight='bold')
ax.legend(title='Risk Class', bbox_to_anchor=(1.01, 1), loc='upper left')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')

for container in ax.containers:
    ax.bar_label(container, fmt='%.1f%%', label_type='center',
                 fontsize=7, color='white', fontweight='bold')

plt.tight_layout()
plt.savefig('notebooks/eda_plots/03_risk_class_distribution.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_risk_class_distribution.png")


# ── Plot 4 — Correlation: Marikina Youth Camp rainfall vs Sto Nino WL ─────────
print("Generating Plot 4: Rainfall vs Water Level correlation...")

sto_nino = wl[wl['station_name'] == 'Sto Nino'][['datetime', 'water_level']].copy()
youth_camp = rf[rf['station_name'] == 'Marikina (Youth Camp)'][
    ['datetime', 'rainfall_mm']].copy()

merged = pd.merge(sto_nino, youth_camp, on='datetime', how='inner')
merged = merged.sort_values('datetime')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

ax1.bar(merged['datetime'], merged['rainfall_mm'],
        width=0.04, color='steelblue', alpha=0.7, label='Rainfall (mm/hr)')
ax1.set_ylabel('Rainfall (mm/hr)', fontsize=10)
ax1.set_title('Marikina (Youth Camp) Rainfall', fontsize=11)
ax1.legend(fontsize=9)

ax2.plot(merged['datetime'], merged['water_level'],
         color='steelblue', linewidth=0.8)
ax2.axhline(15.0, color='orange', linestyle='--', linewidth=1.2,
            label='Alert (15m)')
ax2.axhline(17.0, color='red',    linestyle='--', linewidth=1.2,
            label='Critical (17m)')
ax2.set_ylabel('Water Level (m)', fontsize=10)
ax2.set_title('Sto Nino Water Level', fontsize=11)
ax2.legend(fontsize=9)
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

plt.suptitle('Rainfall vs Water Level: Youth Camp → Sto Nino',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('notebooks/eda_plots/04_rainfall_vs_waterlevel.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_rainfall_vs_waterlevel.png")


# ── Plot 5 — Water level distribution boxplot per station ─────────────────────
print("Generating Plot 5: Water level distribution boxplot...")

fig, ax = plt.subplots(figsize=(14, 6))
wl_sorted = wl.copy()
wl_sorted['station_name'] = pd.Categorical(
    wl_sorted['station_name'], categories=STATIONS, ordered=True
)

sns.boxplot(data=wl_sorted, x='station_name', y='water_level',
            palette='Blues', ax=ax)

for station in STATIONS:
    s = wl[wl['station_name'] == station]
    alertwl    = s['alertwl'].iloc[0]
    criticalwl = s['criticalwl'].iloc[0]
    idx = STATIONS.index(station)
    ax.plot([idx - 0.4, idx + 0.4], [alertwl, alertwl],
            color='orange', linewidth=2, linestyle='--')
    ax.plot([idx - 0.4, idx + 0.4], [criticalwl, criticalwl],
            color='red', linewidth=2, linestyle='--')

ax.set_xlabel('')
ax.set_ylabel('Water Level (m)', fontsize=11)
ax.set_title('Water Level Distribution per Station\n'
             '(Orange dashed = Alert, Red dashed = Critical)',
             fontsize=12, fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
plt.tight_layout()
plt.savefig('notebooks/eda_plots/05_waterlevel_boxplot.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_waterlevel_boxplot.png")


# ── Plot 6 — Monthly average water level heatmap ──────────────────────────────
print("Generating Plot 6: Monthly average water level heatmap...")

wl['month']   = wl['datetime'].dt.to_period('M')
monthly_avg   = (wl.groupby(['station_name', 'month'])['water_level']
                   .mean()
                   .reset_index())
monthly_avg['month_str'] = monthly_avg['month'].astype(str)
pivot_heatmap = monthly_avg.pivot(index='station_name',
                                   columns='month_str',
                                   values='water_level')

fig, ax = plt.subplots(figsize=(20, 5))
sns.heatmap(pivot_heatmap, ax=ax, cmap='YlOrRd',
            linewidths=0.3, linecolor='white',
            cbar_kws={'label': 'Avg Water Level (m)'})
ax.set_title('Monthly Average Water Level per Station (Jul 2024 – Jun 2026)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
plt.tight_layout()
plt.savefig('notebooks/eda_plots/06_monthly_heatmap.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 06_monthly_heatmap.png")


# ── Summary statistics ─────────────────────────────────────────────────────────
print()
print("=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print()
print("Water Level per Station:")
print(wl.groupby('station_name')['water_level']
      .agg(['count', 'mean', 'std', 'min', 'max'])
      .round(2).to_string())

print()
print("Rainfall per Station (mm/hr):")
print(rf.groupby('station_name')['rainfall_mm']
      .agg(['count', 'mean', 'std', 'min', 'max'])
      .round(2).to_string())

print()
print("Risk class counts across all stations:")
print(wl['risk_class'].value_counts().to_string())

print()
print("Max water levels recorded:")
print(wl.groupby('station_name')['water_level'].max()
      .sort_values(ascending=False).round(2).to_string())

print()
print("=" * 60)
print(f"All plots saved to: notebooks/eda_plots/")
print("EDA COMPLETE")
print("=" * 60)