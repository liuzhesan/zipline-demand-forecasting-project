# Drone Medical Supply Demand Forecasting

Predicts next-week product demand for each health facility served by drone delivery nests, and outputs a ready-to-use shopping list.

---

## Overview

Zipline's drone nests resupply health facilities with vaccines, blood products, medications, etc. This pipeline forecasts weekly demand at the `nest × facility × product` level so each nest can pre-position the right inventory.

Currently covers four nests across Ghana and Rwanda, but the pipeline auto-discovers data — drop a new CSV into `data/` and it's included automatically.

| Nest | Country | Data Range |
|------|---------|------------|
| GH1 Omenako | Ghana | Jan 2024 – Feb 2026 |
| GH3 Vobsi | Ghana | Jan 2024 – Feb 2026 |
| RW1 Muhanga | Rwanda | Dec 2024 – Apr 2026 |
| RW2 Kayonza | Rwanda | Dec 2024 – Apr 2026 |

---

## Folder Structure

```
FinalDeliverable/
├── README.md
├── 01_Data_Preprocessing.ipynb       <-- raw data → weekly panel + features
├── 02_Demand_Forecasting.ipynb       <-- model training → shopping list
└── data/
    ├── *.csv                         <-- raw delivery files (drop new ones here)
    ├── panel_weekly.parquet          <-- (generated) weekly demand panel
    ├── unified_features.parquet      <-- (generated) 40-feature ML table
    └── facility_shopping_list_next_week.csv  <-- (generated) final output
```

---

## How to Run (Google Colab)

1. Upload the raw delivery CSVs to a folder in your Google Drive.
2. Open `01_Data_Preprocessing.ipynb` in Colab. Set `DATA_DIR` to your Google Drive folder path (e.g. `/content/drive/MyDrive/Project/data`). Run all cells.
3. Open `02_Demand_Forecasting.ipynb` in Colab. Set the same `DATA_DIR` path. Run all cells. The shopping list CSV will auto-download at the end.

**Requirements** (pre-installed on Colab): `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `xgboost`.

---

## Adding New Data

1. Upload the new CSV to the same Google Drive folder. It needs columns: `DATE`, `NEST_NAME`, `SHIPMENT_PRIORITY`, `HEALTH_FACILITY_NAME`, `UNITS_DELIVERED`, `PRODUCT_NAME`. Extra columns are ignored.
2. Re-run both notebooks. The new data is picked up automatically.
3. Files without the required columns are skipped with a warning.

---

## Pipeline Design

### Notebook 01 — Data Preprocessing

1. **Load & clean**: Auto-discover CSVs, drop nulls, filter to planned shipments only (exclude unpredictable `emergency` orders).
2. **Weekly aggregation**: Monday-aligned weeks, zero-filled per nest (each nest has its own date range).
3. **Calendar split** (per nest): last 5 weeks → test, prior 5 → validation, rest → train.
4. **Demand classification** (Syntetos-Boylan): ADI × CV² → Smooth / Erratic / Intermittent / Lumpy.
5. **Tier labelling**: pairs with ≥5 nonzero weeks, ADI ≤ 15, active in last 26 weeks → `modellable` (ML). Everything else → `rare` (heuristic).
6. **Feature engineering**: 40 past-only features — calendar, lags, rolling stats, intermittent signals, expanding history, label encodings.

### Notebook 02 — Demand Forecasting

**Model**: Per-demand-class XGBoost, two-stage with hard-threshold gating.

- **Stage 1**: classifier predicts P(demand > 0), with class-imbalance correction.
- **Stage 2**: regressor on log1p(units), trained on positive rows only.
- **Combination**: if P ≥ threshold → regressor output, else 0. Threshold tuned per class on validation using asymmetric WAPE (3× under-prediction penalty).

Why per-class? Smooth pairs depend on recent lags; Intermittent pairs depend on timing features. A single model can't serve both well.

**Rare pairs** use `mean_nonzero × nonzero_rate` — the long-run expected weekly demand.

**Production run**: retrain on all data, recompute features with full history, score next week, combine ML + heuristic → shopping list.

---

## Reading the Shopping List

`facility_shopping_list_next_week.csv` — one row per (nest, facility, product), sorted by predicted demand within each facility.

| Column | Meaning |
|--------|---------|
| `nest`, `facility`, `product` | What and where |
| `week_start` | Week being predicted |
| `predicted_units` | Continuous prediction |
| `predicted_units_rounded` | Integer-rounded for ordering |
| `tier` | `modellable` (ML) or `rare` (heuristic) |
| `demand_class` | Smooth / Erratic / Intermittent / Lumpy |
| `method` | Which method produced it |
| `s1_prob`, `s2_mag` | Model diagnostics (modellable only) |

---

## Key Design Decisions

- **No emergency orders**: they're unpredictable and would add noise.
- **Per-nest splits**: GH and RW have different date ranges, so boundaries are set independently.
- **Hard threshold > soft**: avoids spurious small predictions on true-zero rows.
- **3× under-prediction penalty**: stock-outs of medical supplies are far worse than over-ordering.
- **Parquet for intermediates**: ~50× smaller than CSV, ~10× faster I/O.
