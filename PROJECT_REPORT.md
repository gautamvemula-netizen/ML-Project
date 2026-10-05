# Project Report: Traffic Flow Pattern Analysis Using K-Means and DBSCAN

## Abstract
Traffic conditions vary across road segments and time. This project demonstrates an unsupervised machine-learning workflow that groups traffic observations using K-Means and DBSCAN. The application supports data upload, feature selection, optional standardization, interactive visualization, cluster profiling, and internal clustering metrics. K-Means represents observations using a fixed number of centroid-based clusters, while DBSCAN identifies dense regions and can mark outliers as noise. The bundled data is synthetic and is intended only to demonstrate the software. Real traffic data is required for conclusions about an actual road network.

## 1. Introduction
Traffic-flow analysis can help summarize observations such as vehicle speed, traffic volume, and detector occupancy. Unsupervised clustering is useful when predefined labels are unavailable. This project compares two complementary clustering approaches through a web interface.

## 2. Problem Statement
Given traffic observations represented by numeric features, discover groups with similar traffic characteristics and compare the behavior of K-Means and DBSCAN.

## 3. Objectives
- Build a browser-based traffic clustering application.
- Allow users to upload a CSV and choose numeric features.
- Apply K-Means and DBSCAN to the selected features.
- Visualize clusters and summarize their feature distributions.
- Compare results using suitable internal evaluation metrics.
- Export cluster assignments for further analysis.

## 4. Dataset
The project includes a synthetic illustrative dataset with speed, traffic volume, occupancy, timestamp, and road-segment fields. It contains simulated free-flow, moderate, congested, and incident-like patterns. These labels are only for explaining how the sample was constructed; they are not ground-truth labels used by the clustering algorithms. For a real study, use a documented traffic dataset and describe its source, collection period, units, missing values, and sampling interval.

## 5. Methodology
### 5.1 Preprocessing
The application converts selected features to numeric values, removes rows with missing or infinite values in those features, and optionally applies StandardScaler. Scaling is important when features use different units or ranges.

### 5.2 K-Means
K-Means partitions observations into K clusters by assigning points to the nearest centroid and iteratively updating centroids. The user selects K. The implementation uses a fixed random seed and multiple initializations for repeatability and stability.

### 5.3 DBSCAN
DBSCAN forms clusters from dense neighborhoods using `eps` and `min_samples`. It can identify irregularly shaped clusters and label outliers as `-1`. Its results are sensitive to parameter selection and feature scaling.

### 5.4 Evaluation
The app reports:
- **Silhouette score:** higher values generally indicate better separation and cohesion.
- **Davies–Bouldin index:** lower values generally indicate more compact, separated clusters.
- **Calinski–Harabasz score:** higher values generally indicate better-defined clusters.
- **Noise percentage:** share of observations labelled as DBSCAN noise.

Metrics are shown only when their mathematical requirements are satisfied. DBSCAN metrics exclude noise points, so compare scores cautiously.

## 6. System Design
1. User selects or uploads a dataset.
2. User selects numeric features and scaling preference.
3. User chooses K-Means, DBSCAN, or both.
4. The application fits the model(s).
5. The interface displays metrics, plots, and cluster profiles.
6. The user downloads the observations with cluster labels.

## 7. Results and Discussion
Run the app and record your actual outputs here. Include:
- Dataset source and number of observations used.
- Selected features and whether scaling was enabled.
- K-Means K value and DBSCAN parameters.
- Metric values shown by the app.
- Screenshots of the plots and cluster profile.
- A reasoned interpretation of what each cluster represents.

Do not report invented performance values. Results depend on the dataset and chosen parameters.

## 8. Conclusion
The application provides an interactive comparison of centroid-based and density-based clustering for traffic observations. K-Means is useful when the desired number of groups is known and clusters are reasonably compact. DBSCAN can identify dense groups and noise without a preset cluster count. Reliable real-world conclusions require real traffic data, careful parameter selection, and domain validation.

## 9. Future Enhancements
- Integrate a verified public traffic dataset or live sensor feed.
- Add time-of-day and road-segment filtering.
- Use DBSCAN parameter diagnostics such as a k-distance plot.
- Add geographic maps when latitude/longitude data is available.
- Evaluate stability across time periods and road segments.
- Add a supervised model only if trustworthy labelled data is available.

## 10. Course Outcome Mapping
- **CO1:** select an algorithmic strategy based on the problem structure; compare centroid-based and density-based clustering.
- **CO6:** discuss reproducibility and initialization in K-Means; an extension could benchmark execution time. The current project does not implement parallel algorithms.
- CO2, CO3, CO4, and CO5 in the supplied table concern string processing, dynamic programming, network flow, and NP-completeness/approximation; they are not direct learning outcomes of this project.

## References
1. J. MacQueen, “Some Methods for Classification and Analysis of Multivariate Observations,” 1967.
2. M. Ester, H.-P. Kriegel, J. Sander, and X. Xu, “A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise,” 1996.
3. scikit-learn documentation: Clustering, KMeans, DBSCAN, and clustering performance evaluation.
4. Streamlit documentation: Build and deploy data apps.
