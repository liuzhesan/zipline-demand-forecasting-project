# Unified Pipeline — Facility-level Weekly Demand Forecasting

A clean re-implementation of the demand-forecasting pipeline.
**Goal**: for every (`nest` × `health_facility` × `product`), predict next week's expected demand and assemble a per-facility shopping list.

The four notebooks below are meant to be read top-to-bottom. There are **no `.py` modules** — every operation lives inside a notebook cell.

---

## 1. Folder layout

```
Unified Pipeline/
├── README.md                                         <-- this file
├── 01_Data_Prep_and_Unified_Features.ipynb           <-- single source of truth
├── 02_Statistical_Baseline.ipynb                     <-- per-quadrant Croston / SBA / ETS
├── 03_ML_Models_Unified.ipynb                        <-- LightGBM + XGBoost (same features)
├── 04_Comparison_and_Shopping_List.ipynb             <-- comparison + production deliverable
└── data/
    ├── GH-1 2024 2025 2026 Demand.csv                <-- raw GH-1 deliveries
    ├── GH-3 2024 2025 2026 Demand.csv                <-- raw GH-3 deliveries
    ├── RW1 Demand.csv                                <-- raw RW1 deliveries
    └── RW2 Demand.csv                                <-- raw RW2 deliveries
```

After running the notebooks, the following artifacts are produced under `data/`:

| File | Produced by | What it is |
|---|---|---|
| `panel_weekly.parquet` | 01 | Monday-aligned weekly panel for **every** observed (nest, facility, product). Columns: pair keys, `week_start`, `weekly_units`, `demand_class`, `tier`, `split`, `mean_nonzero`, `nonzero_weeks`. |
| `unified_features.parquet` | 01 | Supervised feature table for `tier == 'modellable'` pairs only. 40 features shared by both ML models. |
| `stat_predictions.csv` | 02 | Croston / SBA / ETS predictions on test rows. |
| `lgbm_predictions.csv` | 03 | LightGBM two-stage predictions on test rows. |
| `xgb_predictions.csv` | 03 | XGBoost two-stage predictions on test rows. |
| `metrics.csv` | 04 | Overall + per-nest + per-class + per-week metrics for all 3 models. |
| `evaluation_plots/01–04_*.png` | 04 | Comparison plots (overall, by nest, by class, by week). |
| `facility_shopping_list_next_week.csv` | 04 | **Final deliverable** — per facility × product, predicted units for the upcoming week. |

> Parquet is used for the heavy intermediate files (panel ~2.5M rows, features ~340k rows) because it is roughly 50× smaller and 10× faster to read/write than CSV. The final shopping list is CSV for easy inspection.

---

## 2. Run order

Run notebooks 01 → 02 → 03 → 04 in the same Python kernel.

| # | Notebook | Reads | Writes | Runtime |
|---|----------|-------|--------|---------|
| 1 | `01_Data_Prep_and_Unified_Features.ipynb` | 4 raw CSVs | `panel_weekly.parquet`, `unified_features.parquet` | ~17 s |
| 2 | `02_Statistical_Baseline.ipynb` | `panel_weekly.parquet` | `stat_predictions.csv` | ~5 s |
| 3 | `03_ML_Models_Unified.ipynb` | `unified_features.parquet` | `lgbm_predictions.csv`, `xgb_predictions.csv` | ~3 s |
| 4 | `04_Comparison_and_Shopping_List.ipynb` | all of the above | `metrics.csv`, plots, `facility_shopping_list_next_week.csv` | ~14 s |

Total ≈ 40 seconds end-to-end on a laptop.

---

## 3. Conventions used everywhere

- **Grain**: `NEST_NAME × HEALTH_FACILITY_NAME × PRODUCT_NAME × week_start` (Monday-aligned).
- **Priority filter**: only "planned" deliveries kept (`custom window`, `resupply`, `scheduled`, `fast`). Reactive `emergency` is dropped.
- **Per-nest calendar split**: each nest's last 5 weeks → `test`, prior 5 weeks → `valid`, rest → `train`. We split per-nest because GH and RW have different end dates (GH ends ~2026-02-11, RW ends ~2026-04-01).
- **Universe**: every observed (nest × facility × product) is in `panel_weekly.parquet`. Within that, each pair is tagged as:
  - `modellable`: `nonzero_weeks ≥ 5` AND `ADI ≤ 15` AND active in last 26 weeks. ML models train on these.
  - `rare`: everything else. The shopping list uses a transparent heuristic for these.
- **Target convention**: predict `weekly_units` of week W using only information available at the end of week W-1 (lags / rolling / history are all shifted accordingly).
- **No `category` / `subcategory` features**: per the agreed plan, only `HEALTH_FACILITY_NAME` and `PRODUCT_NAME` are label-encoded.

---

## 4. Modelling design

### Statistical baseline (Notebook 02)

| `demand_class` | Method | Why |
|---|---|---|
| Smooth / Erratic | ETS (additive trend, no seasonality) | Regular demand → simple level model |
| Intermittent | Croston (α tuned on training tail) | Many zeros, fairly stable order size |
| Lumpy | SBA (bias-corrected Croston) | Many zeros and volatile order size |

### ML models (Notebook 03) — same architecture for both

- Stage 1: occurrence classifier `P(units > 0)` with class-imbalance correction.
- Stage 2: magnitude regressor on `log1p(units)`, trained on positive rows only.
- Final prediction = `P × E[units|positive]` (soft combination, no hard threshold).

We use a soft combination because >75% of validation rows are true zeros, so any hard-gate threshold either fires too often (over-shoots WAPE) or never fires (predicts all zeros). The soft form gives a non-zero expected demand for every (facility, product) pair, which is what the shopping list needs.

The old high/medium/low segmentation from the previous codebase is gone — it complicated the comparison without clear benefit.

### Comparison + shopping list (Notebook 04)

Backtest comparison: outer-join the three prediction tables on `(pair, week)` so every model is scored on the same rows. Compute MAE / RMSE / Bias / WAPE / sMAPE overall, by nest (4 groups), by demand class, and by week. Save plots to `evaluation_plots/`.

Production: pick the champion ML model (lowest overall test WAPE), retrain on all data (train + valid + test), and score one new week per nest at `nest_max_week + 7d`. Modellable pairs use the model; rare pairs use `historical_mean_nonzero × historical_nonzero_rate`. Final output: `facility_shopping_list_next_week.csv`, sorted within each facility by predicted units.

---

## 5. Configuration knobs (live in Notebook 01)

| Setting | Default | Meaning |
|---|---|---|
| `PRIORITY_SCOPE` | `['custom window','resupply','scheduled','fast']` | Shipment priorities kept |
| `ADI_CUTOFF`, `CV2_CUTOFF` | `1.32`, `0.49` | Syntetos–Boylan thresholds |
| `MIN_NONZERO_WEEKS` | `5` | Minimum non-zero weeks for `modellable` |
| `MAX_ADI` | `15.0` | Maximum ADI for `modellable` |
| `RECENCY_WEEKS` | `26` | Pair must be active within last N weeks |
| `N_TEST_WEEKS`, `N_VALID_WEEKS` | `5`, `5` | Per-nest calendar split sizes |

---

## 6. Environment

Python 3.10+ with: `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `statsmodels`, `lightgbm`, `xgboost`, `matplotlib`, `seaborn`. Prophet is **not** required.

---

## 7. Reading the shopping list

`facility_shopping_list_next_week.csv` columns:

| column | meaning |
|---|---|
| `nest`, `facility`, `product` | pair identifier |
| `week_start` | the upcoming week being predicted (per-nest, `nest_max_week + 7d`) |
| `predicted_units` | model expected demand (continuous) |
| `predicted_units_rounded` | integer-rounded demand for ordering |
| `tier` | `modellable` (ML) or `rare` (heuristic) |
| `demand_class` | ADI-CV² class (Smooth / Erratic / Intermittent / Lumpy) |
| `method` | which model produced the prediction |
| `s1_prob`, `s2_mag` | model diagnostics (only for `modellable`) |

Rows are sorted by `(nest, facility, predicted_units desc)` so reading row-by-row within a facility gives that facility's shopping list ranked by expected demand.
