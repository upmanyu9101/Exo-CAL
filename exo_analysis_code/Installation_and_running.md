# 🪐 Exoplanet Candidate Analysis Toolkit

A Python-based pipeline for analyzing **TESS (TOI)**, **Kepler (KOI)**, and **K2** exoplanet candidate datasets.  
This toolkit performs feature extraction, probabilistic classification, and visualization of potential exoplanet candidates using reproducible ML workflows.

---

## 🚀 Features

- Automated ingestion of TOI, KOI, and K2 datasets (CSV format)  
- Data cleaning, feature standardization, and imputation  
- ML-based candidate scoring with probability, entropy, and calibrated uncertainty  
- Cross-validation and performance metrics (ROC-AUC, PR-AUC)  
- PCA-based visualizations and per-target summaries  
- Active learning support to identify uncertain or ambiguous candidates  
- Reproducible results with deterministic seeding

---

## ⚙️ Requirements

- **Python:** 3.9–3.12  
- **Git:** optional (for cloning)  
- **Disk space:** ~1–2 GB for outputs  
- **CSV input files:**
  - **TOI (TESS):** `toi`, `tfopwg_disp`, `pl_orbper`, `pl_trandurh`, `pl_trandep`, `pl_rade`, `st_teff`, `st_rad`, etc.  
  - **KOI (Kepler):** `kepoi_name`, `koi_disposition`, `koi_period`, `koi_duration`, `koi_depth`, `koi_prad`, `koi_steff`, `koi_srad`, etc.  
  - **K2:** `pl_name`, `disposition`, `pl_orbper`, `pl_trandurh`, `pl_trandep`, `pl_rade`, `st_teff`, `st_rad`, etc.  

> Comments starting with `#` in CSV files are automatically skipped.

---

## 🧩 Installation

### 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -U pip
pip install -r requirements.txt
```

---

## ▶️ Running the Analysis

**Basic command:**
```bash
python exo_analysis.py --toi "<PATH_TO_TOI_CSV>" --koi "<PATH_TO_KOI_CSV>" --k2 "<PATH_TO_K2_CSV>" --out "<OUTPUT_DIR>"
```

**Example (Windows):**
```powershell
python .\exo_analysis.py `
  --toi "C:\data\toi.csv" `
  --koi "C:\data\koi.csv" `
  --k2  "C:\data\k2.csv" `
  --out "C:\MATLAB\TOI_analysis\out"
```

**Example (macOS / Linux):**
```bash
python exo_analysis.py   --toi "/data/toi.csv"   --koi "/data/koi.csv"   --k2  "/data/k2.csv"   --out "/Users/<you>/ExoRuns/out"
```

**Optional flags:**
| Flag | Description |
|------|--------------|
| `--limit-targets N` | Run only the first *N* entries (for testing) |
| `--seed 42` | Ensure reproducibility |
| `--quiet` | Suppress console logs |

---

## 📊 Outputs

Each dataset produces results under `<OUTPUT_DIR>/<DATASET>/`:

| File | Description |
|------|--------------|
| `dataset_predictions.csv` | Predictions with probabilities, entropy, and calibration fields |
| `top_candidates.csv` | Sorted top candidates |
| `active_learning_queue.csv` | Ambiguous / high-entropy targets |
| `threshold_table.csv` | Precision–recall tradeoffs |
| `feature_importance.csv` | Global feature importances |
| `perf.csv` | Performance metrics (ROC-AUC, PR-AUC, etc.) |
| `fig/` | Figures (ROC, PR, reliability, PCA) |
| `targets/<designation>/` | Per-target reports: `report.json`, `report.png` |

---

## 🖼️ Viewing Results

- Open dataset-level figures from `<OUTPUT_DIR>/<DATASET>/fig/`
- Browse per-target visualizations in `<OUTPUT_DIR>/<DATASET>/targets/`

---

## 🧰 Troubleshooting

| Issue | Fix |
|-------|-----|
| **Windows paths not recognized** | Always wrap in double quotes (`"C:\path\file.csv"`) |
| **NaN warnings** | Check column names; script supports NASA and MATLAB headers |
| **KOI only has positives** | Expected: fallback to one-class scoring mode |
| **Matplotlib errors (headless)** | Set backend: `export MPLBACKEND=Agg` or `$env:MPLBACKEND="Agg"` |

---

## 🔁 Clean Re-run
    
To restart the analysis:
```bash
rm -rf <OUTPUT_DIR>
python exo_analysis.py --toi ... --koi ... --k2 ... --out ...
```

---

## ❌ Deactivate Environment

```bash
deactivate
```

---

## 💬 Support

If you encounter issues:
1. Copy the full command and last 30 console lines  
2. Include:
   - OS and Python version  
   - `pip list` output inside venv  
   - First 3 rows of your CSV (sanitized)

Open an issue on the repository and include this info.

---

**Happy planet hunting! 🪐**
