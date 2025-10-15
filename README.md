# ML2 Semestral Project — Starter

This repo scaffold helps you complete the **Over/Under 2.5 goals** project cleanly and reproducibly.

## Setup
1. Create a virtual env and install deps:
   ```bash
   pip install -r requirements.txt
   ```
2. Unzip the provided `data.zip` and place its folder contents into:
   ```
   data/raw/
   ```

## Run order (notebooks)
1. `01_data_overview.ipynb` — Quick EDA, sanity checks.
2. `02_data_prep_baseline.ipynb` — Clean/merge baseline columns; create target; save to `data/processed/`.
3. `03_model_baseline.ipynb` — Train/evaluate baseline pipeline; save metrics/artifacts to `outputs/`.
4. `04_feature_engineering.ipynb` — Build extended features; persist datasets.
5. `05_model_extended.ipynb` — Train/evaluate extended pipeline; save metrics/artifacts.
6. `06_error_analysis.ipynb` — Analyze misclassifications by league/season.
7. `07_profit_optimization.ipynb` — Compute optimal profit margin and ΔΠ per market.
8. `08_summary_report.ipynb` — Clean narrative & tables for submission.

> Keep heavy logic in `src/` so notebooks stay readable. Parameters live in `config/config.yaml`.

## Scripts (optional)
You can also run steps headlessly:
```bash
python scripts/make_processed.py
python scripts/train_baseline.py
python scripts/train_extended.py
python scripts/compute_profit.py
```

## Config
Edit **config/config.yaml** once; all notebooks/scripts read it.