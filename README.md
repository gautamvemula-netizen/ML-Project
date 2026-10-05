# Traffic Flow Pattern Analysis Using K-Means and DBSCAN

A complete, runnable Streamlit machine-learning web application for exploring traffic observations and comparing K-Means and DBSCAN clustering.

## Features
- Included illustrative sample dataset (`data/traffic_sample.csv`)
- Upload your own traffic CSV
- Select numeric features and optionally standardize them
- Run K-Means, DBSCAN, or compare both
- Interactive scatter plots and cluster profiles
- Silhouette, Davies–Bouldin, and Calinski–Harabasz metrics (when mathematically defined)
- Download the dataset with cluster labels

> **Data note:** the bundled dataset is synthetic/illustrative. It is not live traffic data. Replace it with real sensor/traffic data for a real-world study.

## 1. Requirements
- Python 3.10 or newer
- VS Code
- Python extension for VS Code

## 2. Run locally in VS Code

Open this folder in VS Code. In the terminal:

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

### Windows PowerShell
```powershell
py -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## 3. Deploy it so others can access the website
One straightforward option is Streamlit Community Cloud:
1. Create a GitHub account (if needed) and create a **public** repository.
2. Upload the contents of this project folder to that repository.
3. Go to https://share.streamlit.io/ and sign in with GitHub.
4. Create a new app, select the repository and branch, and set the main file path to `app.py`.
5. Deploy. Streamlit will provide a public URL you can share.

Do not upload private datasets, passwords, API keys, or personal information to a public repository. Deployment options and account requirements can change; follow the current instructions on Streamlit's site.

## Input CSV format
The app accepts any CSV with at least two numeric columns. Recommended fields:

| Column | Meaning | Example |
|---|---|---:|
| `speed_kmh` | Average vehicle speed in km/h | 42.5 |
| `traffic_volume` | Vehicle count per time interval | 61 |
| `occupancy` | Fraction of detector time occupied, from 0 to 1 | 0.32 |
| `timestamp` | Optional time | `2026-01-01 08:00:00` |
| `road_segment` | Optional road identifier | `A1` |

Your own column names can differ. Select the numeric features in the sidebar. Missing/non-numeric values in selected features are skipped.

## Suggested experiment
1. Run with the default features and standardization enabled.
2. Compare K-Means with K = 2, 3, 4, and 5.
3. Adjust DBSCAN `eps` and `min_samples`. If DBSCAN returns one cluster or all noise, change these parameters.
4. Compare the cluster profiles and metric values.
5. Repeat using different feature combinations.
6. Export the results and discuss limitations.

## Project methodology
1. Data loading: bundled CSV or uploaded file.
2. Data cleaning: coerce selected features to numeric; remove rows with missing/infinite selected values.
3. Feature scaling: optional StandardScaler.
4. K-Means: centroid-based partitioning, fixed K, reproducible initialization.
5. DBSCAN: density-based clustering with a noise label of -1.
6. Evaluation: internal clustering metrics where valid.
7. Visualization: scatter plots, cluster summaries, downloadable labels.

## Limitations
- Clustering discovers structure; it does not prove that clusters correspond to true traffic states.
- The sample dataset is synthetic and only demonstrates the workflow.
- A 2D scatter plot displays only two selected features, even if clustering uses more.
- DBSCAN is sensitive to `eps`, `min_samples`, and feature scaling.
- Internal metrics should be interpreted alongside domain knowledge and data quality.

## Project-to-course-outcome mapping (based on the supplied CO table)
- **CO1 (algorithm strategy):** identify the clustering task and choose between centroid-based K-Means and density-based DBSCAN.
- **CO6 (randomized/parallel algorithm concepts):** discuss K-Means initialization/reproducibility and compare runtime/quality across settings as an extension. This project does not implement a parallel algorithm.
- The supplied CO2, CO3, CO4, and CO5 focus on string algorithms, dynamic programming, network flow, and complexity/approximation topics; they are not directly implemented by this clustering project. Confirm with your instructor if a specific CO mapping is required.

## Files
- `app.py` — Streamlit web application
- `data/traffic_sample.csv` — illustrative sample data
- `requirements.txt` — Python dependencies
- `PROJECT_REPORT.md` — report draft you can adapt to your institution's format
