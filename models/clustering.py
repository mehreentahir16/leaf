import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import seaborn as sns

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score

def hybrid_distance_metric(X, Y=None):
    # Ensure X and Y are 2D arrays
    X = np.atleast_2d(X)
    if Y is not None:
        Y = np.atleast_2d(Y)

    # Calculate cosine and Manhattan distances
    cosine_dist = pairwise_distances(X, Y, metric='cosine')
    manhattan_dist = pairwise_distances(X, Y, metric='manhattan')

    # Combine distances
    hybrid_dist = 0.8 * cosine_dist + 0.2 * manhattan_dist
    return hybrid_dist


def run_hdbscan_clustering(gradient_magnitudes, gradient_variances, hardware_scores, network_scores, losses, client_num_samples):
    # Create initial DataFrame
    client_ids = list(gradient_magnitudes.keys())
    features = pd.DataFrame({
        'Num Samples': [client_num_samples[c_id] for c_id in client_ids],
        'Gradient Magnitude': [gradient_magnitudes[c_id] for c_id in client_ids],
        'Gradient Variance': [gradient_variances[c_id] for c_id in client_ids]
    })

    # Standardize Num Samples for Agglomerative Clustering
    scaler = StandardScaler()
    features['Scaled Num Samples'] = scaler.fit_transform(features[['Num Samples']])

    # Step 1: Apply Agglomerative Clustering on Num Samples
    agglomerative_clusterer = AgglomerativeClustering(n_clusters=2, linkage='complete')
    sample_clusters = agglomerative_clusterer.fit_predict(features[['Scaled Num Samples']])

    # Initialize storage for final clusters and metrics
    final_clusters = {}
    all_outlier_scores = []
    all_stability_scores = []
    all_persistence = []
    total_outliers = 0
    silhouette_scores = []
    davies_bouldin_scores = []

    # Step 2: For each Agglomerative cluster, apply HDBSCAN
    for cluster_id in np.unique(sample_clusters):
        cluster_indices = np.where(sample_clusters == cluster_id)[0]
        cluster_client_ids = [client_ids[i] for i in cluster_indices]

        # Prepare sub-DataFrame for each Agglomerative cluster with relevant features
        sub_features = features.iloc[cluster_indices][['Gradient Magnitude', 'Gradient Variance']]
        scaled_sub_features = scaler.fit_transform(sub_features)

        # Run HDBSCAN on the scaled subset
        hdbscan_clusterer = hdbscan.HDBSCAN(
            algorithm='best',
            approx_min_span_tree=False,
            gen_min_span_tree=True,
            metric=hybrid_distance_metric,
            leaf_size=50,
            min_cluster_size=13,
            min_samples=3
        )
        hdbscan_labels = hdbscan_clusterer.fit_predict(scaled_sub_features)

        # Calculate metrics for this HDBSCAN cluster subset if there are more than one cluster and no single cluster
        if len(set(hdbscan_labels)) > 1 and -1 not in hdbscan_labels:
            silhouette = silhouette_score(scaled_sub_features, hdbscan_labels)
            db_score = davies_bouldin_score(scaled_sub_features, hdbscan_labels)
            silhouette_scores.append(silhouette)
            davies_bouldin_scores.append(db_score)

        # Collect metrics from HDBSCAN for each subset
        outlier_scores = hdbscan_clusterer.outlier_scores_
        stability_scores = hdbscan_clusterer.probabilities_
        persistence = hdbscan_clusterer.cluster_persistence_
        n_outliers = (hdbscan_labels == -1).sum()

        # Store clustering results and metrics
        for i, c_id in enumerate(cluster_client_ids):
            final_clusters[c_id] = hdbscan_labels[i] if hdbscan_labels[i] != -1 else -1

        all_outlier_scores.extend(outlier_scores)
        all_stability_scores.extend(stability_scores)
        all_persistence.extend(persistence)
        total_outliers += n_outliers

    print("Cluster Labels:", list(final_clusters.values()))
    print("Outlier Scores:", all_outlier_scores)
    print("Stability Scores:", all_stability_scores)
    print("Persistence:", all_persistence)
    print("Number of Outliers Detected:", total_outliers)
    print("Average Silhouette Score (for valid clusters):", np.mean(silhouette_scores) if silhouette_scores else "N/A")
    print("Average Davies-Bouldin Index (for valid clusters):", np.mean(davies_bouldin_scores) if davies_bouldin_scores else "N/A")

    # Visualization with pairplot
    features = features.drop(columns=['Scaled Num Samples'])
    features['Cluster'] = [final_clusters[c_id] for c_id in client_ids]
    sns.pairplot(features, hue='Cluster', palette='tab10', plot_kws={'alpha': 0.6})
    plt.suptitle("Pair Plot of Clusters (Agglomerative + HDBSCAN)", y=1.02)
    plt.show()

    return final_clusters

def set_baseline_ranges(client_clusters, gradient_magnitudes, gradient_variances, num_epochs, adjustment_factor=1.1):
    baseline_ranges = {}
    for cluster_id in np.unique(list(client_clusters.values())):
        if cluster_id == -1:  # Skip outliers
            continue
        cluster_clients = [c_id for c_id, c in client_clusters.items() if c == cluster_id]
        magnitudes = np.array([gradient_magnitudes[c_id] for c_id in cluster_clients])
        variances = np.array([gradient_variances[c_id] for c_id in cluster_clients])

        # Compute initial range with conditional adjustment for epoch variation
        magnitude_range = (magnitudes.min(), magnitudes.max())
        variance_range = (variances.min(), variances.max())

        if num_epochs > 1:
            magnitude_range = (magnitude_range[0] * adjustment_factor, magnitude_range[1] * adjustment_factor)
            variance_range = (variance_range[0] * adjustment_factor, variance_range[1] * adjustment_factor)

        baseline_ranges[cluster_id] = {
            'magnitude_range': magnitude_range,
            'variance_range': variance_range
        }
    return baseline_ranges
