import io
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.preprocessing import StandardScaler

DEFAULT_PATH = Path(__file__).parent / "data" / "traffic_sample.csv"


@st.cache_data
def load_sample():
    if not DEFAULT_PATH.exists():
        DEFAULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        demo_df = make_demo_data()
        demo_df.to_csv(DEFAULT_PATH, index=False)
        return demo_df
    return pd.read_csv(DEFAULT_PATH)


def make_demo_data(n=900, seed=42):
    """Create a reproducible illustrative dataset when a user explicitly requests demo mode."""
    rng = np.random.default_rng(seed)
    groups = [
        # label, count, speed mean/sd, volume mean/sd, occupancy mean/sd
        ("Free-flow", int(n * .38), 72, 8, 24, 7, 0.12, 0.04),
        ("Moderate", int(n * .34), 48, 7, 58, 12, 0.31, 0.07),
        ("Congested", int(n * .22), 19, 6, 92, 15, 0.69, 0.10),
        ("Incident-like", n - int(n * .38) - int(n * .34) - int(n * .22), 12, 10, 28, 13, 0.48, 0.17),
    ]
    rows = []
    timestamp = pd.date_range("2026-01-01 06:00", periods=n, freq="5min")
    idx = 0
    for name, count, sm, ss, vm, vs, om, osd in groups:
        for _ in range(count):
            rows.append({
                "timestamp": timestamp[idx],
                "speed_kmh": max(0, rng.normal(sm, ss)),
                "traffic_volume": max(0, rng.normal(vm, vs)),
                "occupancy": float(np.clip(rng.normal(om, osd), 0, 1)),
                "road_segment": rng.choice(["A1", "A2", "B1", "B2"]),
                "illustrative_state": name,
            })
            idx += 1
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df


def clean_numeric(df, features):
    x = df[features].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    valid = x.notna().all(axis=1)
    return x.loc[valid].copy(), valid


def main():
    st.set_page_config(page_title="Traffic Flow Pattern Analysis", page_icon="🚦", layout="wide")

    st.title("🚦 Traffic Flow Pattern Analysis Using K-Means and DBSCAN")
    st.markdown(
        "Explore traffic observations, discover recurring traffic patterns, and compare "
        "two unsupervised machine-learning algorithms."
    )

    data_source = "Included sample dataset"
    raw = None

    with st.sidebar:
        st.header("1. Data")
        uploaded = st.file_uploader("Upload traffic CSV", type=["csv"])
        use_demo = st.checkbox(
            "Use demo data for testing",
            value=False,
            help="Optional: load the built-in sample data instead of your uploaded CSV."
        )

        if uploaded is not None:
            try:
                raw = pd.read_csv(uploaded)
                data_source = "Uploaded CSV"
            except Exception as e:
                st.error(f"Could not read CSV: {e}")
                st.stop()
        elif use_demo:
            try:
                raw = load_sample()
                data_source = "Included sample dataset"
            except Exception:
                st.info("Upload a traffic CSV to begin the analysis.")
                st.stop()
        else:
            st.info("Upload a traffic CSV to begin the analysis.")
            st.stop()

        st.caption(f"Source: {data_source}")
        st.header("2. Features")
        numeric_cols = raw.select_dtypes(include=np.number).columns.tolist()
        preferred = [c for c in ["speed_kmh", "traffic_volume", "occupancy"] if c in numeric_cols]
        default_features = preferred if len(preferred) >= 2 else numeric_cols[:min(3, len(numeric_cols))]
        features = st.multiselect(
            "Select at least 2 numeric features",
            options=numeric_cols,
            default=default_features
        )
        algorithm = st.radio("Clustering algorithm", ["Compare both", "K-Means", "DBSCAN"])
        scale_data = st.checkbox("Standardize features (recommended)", value=True)

    if len(features) < 2:
        st.warning("Select at least two numeric columns in the sidebar.")
        st.stop()

    X_df, valid_mask = clean_numeric(raw, features)
    if len(X_df) < 5:
        st.error("At least 5 rows with valid numeric values are required.")
        st.stop()

    dropped = len(raw) - len(X_df)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df) if scale_data else X_df.to_numpy()

    st.caption(f"Rows: {len(raw):,} · Rows used: {len(X_df):,} · Invalid rows skipped: {dropped:,}")
    with st.expander("Preview data"):
        st.dataframe(raw.head(20), width="stretch")
        st.download_button("Download input data as CSV", raw.to_csv(index=False).encode("utf-8"),
                           "traffic_input.csv", "text/csv")

    c1, c2, c3 = st.columns(3)
    c1.metric("Observations used", f"{len(X_df):,}")
    c2.metric("Selected features", len(features))
    c3.metric("Features scaled", "Yes" if scale_data else "No")

    st.subheader("Traffic feature relationships")
    if len(features) >= 2:
        plot_df = X_df.copy()
        for column in ["timestamp", "road_segment"]:
            if column in raw.columns:
                plot_df[column] = raw.loc[X_df.index, column].to_numpy()
        fig = px.scatter(
            plot_df, x=features[0], y=features[1],
            color="road_segment" if "road_segment" in plot_df.columns else None,
            hover_data=[c for c in ["timestamp", "road_segment"] if c in plot_df.columns],
            title=f"{features[0]} vs {features[1]}"
        )
        st.plotly_chart(fig, width="stretch")

    st.sidebar.header("3. Model settings")
    k = st.sidebar.slider("K-Means clusters (K)", 2, min(10, max(2, len(X_df)-1)), 3)
    eps = st.sidebar.slider("DBSCAN eps", 0.1, 3.0, 0.65, 0.05,
                            help="Distance threshold in the feature space. If standardized, this is in standard-deviation units.")
    min_samples = st.sidebar.slider("DBSCAN min_samples", 2, 30, 8)

    def evaluate(labels, data):
        labels = np.asarray(labels)
        non_noise = labels != -1
        unique = np.unique(labels[non_noise])
        n_clusters = len(unique)
        noise_pct = float(np.mean(labels == -1) * 100)
        result = {
            "Clusters (excluding noise)": n_clusters,
            "Noise points (%)": round(noise_pct, 2),
            "Silhouette": np.nan,
            "Davies–Bouldin": np.nan,
            "Calinski–Harabasz": np.nan,
        }
        mask = non_noise
        if n_clusters >= 2 and mask.sum() > n_clusters:
            try:
                result["Silhouette"] = round(float(silhouette_score(data[mask], labels[mask])), 4)
                result["Davies–Bouldin"] = round(float(davies_bouldin_score(data[mask], labels[mask])), 4)
                result["Calinski–Harabasz"] = round(float(calinski_harabasz_score(data[mask], labels[mask])), 2)
            except Exception:
                pass
        return result

    results = {}
    models = {}
    if algorithm in ["Compare both", "K-Means"]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        results["K-Means"] = km.fit_predict(X)
        models["K-Means"] = km
    if algorithm in ["Compare both", "DBSCAN"]:
        db = DBSCAN(eps=eps, min_samples=min_samples)
        results["DBSCAN"] = db.fit_predict(X)
        models["DBSCAN"] = db

    st.subheader("Clustering results")
    metric_rows = []
    for name, labels in results.items():
        m = evaluate(labels, X)
        m["Algorithm"] = name
        metric_rows.append(m)
    metrics_df = pd.DataFrame(metric_rows).set_index("Algorithm")
    st.dataframe(metrics_df, width="stretch")

    if len(results) == 2:
        st.info(
            "Metric guide: higher Silhouette and Calinski–Harabasz values, and lower "
            "Davies–Bouldin values, often indicate better-separated clusters. "
            "DBSCAN marks outliers as -1; its scores exclude those noise points. "
            "Metrics are not directly definitive when cluster shapes differ."
        )

    out = raw.loc[X_df.index].copy()
    for name, labels in results.items():
        out[f"{name.lower().replace('-', '').replace(' ', '_')}_cluster"] = labels

    plot_features = features[:2]
    for name, labels in results.items():
        plot_df = X_df.copy()
        plot_df["Cluster"] = pd.Series(labels, index=plot_df.index).astype(str)
        plot_df["Cluster"] = plot_df["Cluster"].replace({"-1": "Noise (-1)"})
        fig = px.scatter(
            plot_df, x=plot_features[0], y=plot_features[1], color="Cluster",
            title=f"{name}: discovered traffic groups",
            hover_data=features
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Cluster profile")
    selected_name = st.selectbox("Profile algorithm", list(results.keys()))
    profile_df = X_df.copy()
    profile_df["cluster"] = results[selected_name]
    profile = profile_df.groupby("cluster")[features].agg(["count", "mean", "median", "std"]).round(3)
    st.dataframe(profile, width="stretch")

    st.subheader("Download results")
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download observations with cluster labels", csv,
                       "traffic_cluster_results.csv", "text/csv")

    st.markdown("---")
    st.markdown(
        "**How to interpret:** K-Means assigns every observation to one of K centroid-based "
        "groups and works best when groups are roughly compact. DBSCAN groups dense regions "
        "and can label isolated observations as noise; it does not require choosing K. "
        "Results depend on selected features, scaling, and parameters."
    )
    st.caption("Educational project demo. The included sample data is synthetic/illustrative, not live traffic data.")


if __name__ == "__main__":
    main()
