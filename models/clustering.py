# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from sklearn.decomposition import PCA

# from sklearn.cluster import DBSCAN
# from sklearn.preprocessing import StandardScaler


# def run_dbscan_clustering(gradient_magnitudes, gradient_variances, hardware_scores, network_scores, losses, client_num_samples):
#     # Organize features into a matrix
#     client_ids = list(gradient_magnitudes.keys())
#     features = np.array([
#         [
#             gradient_magnitudes[c_id],
#             gradient_variances[c_id],
#             # hardware_scores[c_id],
#             # network_scores[c_id],
#             losses[c_id],
#             # client_num_samples[c_id]
#         ]
#         for c_id in client_ids
#     ])

#     # Standardize features for DBSCAN
#     features = StandardScaler().fit_transform(features)

#     # Set DBSCAN parameters
#     dbscan = DBSCAN(eps=0.5, min_samples=3)
#     cluster_labels = dbscan.fit_predict(features)

#     # Map client IDs to their cluster labels
#     client_clusters = {client_ids[i]: cluster_labels[i] for i in range(len(client_ids))}
    
#     return client_clusters

# def visualize_dbscan_clusters(features, cluster_labels):
#     # Plot the clusters in 3D
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     unique_labels = set(cluster_labels)
#     colors = plt.cm.get_cmap('tab10', len(unique_labels))

#     for label in unique_labels:
#         mask = (cluster_labels == label)
#         ax.scatter(
#             features[mask, 0], features[mask, 1], features[mask, 2],
#             label=f'Cluster {label}',
#             s=50,
#             alpha=0.6,
#             color=colors(label) if label != -1 else 'k'  # Black for outliers
#         )

#     # Set labels
#     ax.set_xlabel('Gradient Variance')
#     ax.set_ylabel('loss')
#     ax.set_zlabel('Number of Samples')
#     ax.set_title("3D Visualization of DBSCAN Clusters")
#     ax.legend()
#     plt.show()

import numpy as np
import hdbscan
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import pairwise_distances
import seaborn as sns

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
    # Combine the features into a DataFrame for easier scaling and visualization
    client_ids = list(gradient_magnitudes.keys())
    features = pd.DataFrame({
        'Gradient Magnitude': [gradient_magnitudes[c_id] for c_id in client_ids],
        'Gradient Variance': [gradient_variances[c_id] for c_id in client_ids],
        'Num Samples': [client_num_samples[c_id] for c_id in client_ids],
        # 'loss': [losses[c_id] for c_id in client_ids]
    })

    # Standardize features for clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Run HDBSCAN with scaled Euclidean distance
    clusterer = hdbscan.HDBSCAN(algorithm='best', approx_min_span_tree=False, gen_min_span_tree=True, metric=hybrid_distance_metric, leaf_size=60, min_cluster_size=15, min_samples=2, p=None)
    cluster_labels = clusterer.fit_predict(scaled_features)

    # Map each client to their cluster
    client_clusters = {client_ids[i]: cluster_labels[i] for i in range(len(client_ids))}

    # Obtain additional metrics
    outlier_scores = clusterer.outlier_scores_
    stability_scores = clusterer.probabilities_
    persistence = clusterer.cluster_persistence_
    n_outliers = (cluster_labels == -1).sum()

    print("Cluster Labels:", cluster_labels)
    print("Outlier Scores:", outlier_scores)
    print("Stability Scores:", stability_scores)
    print("Persistence:", persistence)
    print("Number of Outliers Detected:", n_outliers)

    # # Visualization
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # scatter = ax.scatter(
    #     features['Gradient Magnitude'],
    #     features['Gradient Variance'],
    #     # features['Num Samples'],
    #     c=cluster_labels,
    #     cmap='tab10'
    # )
    # plt.colorbar(scatter, label="Cluster")
    # ax.set_xlabel("Gradient Magnitude")
    # ax.set_ylabel("Gradient Variance")
    # ax.set_zlabel("Num Samples")
    # plt.title("3D Visualization of HDBSCAN Clusters with Scaled Distance")
    # plt.show()
    # fig, ax = plt.subplots(figsize=(10, 7))
    # scatter = ax.scatter(
    #     features['Gradient Magnitude'],
    #     features['Gradient Variance'],
    #     c=cluster_labels,
    #     cmap='tab10'
    # )

    # plt.colorbar(scatter, label="Cluster")
    # ax.set_xlabel("Gradient Magnitude")
    # ax.set_ylabel("Gradient Variance")
    # plt.title("2D Visualization of HDBSCAN Clusters with Scaled Distance")
    # plt.show()
    features['Cluster'] = cluster_labels

    # Using pairplot to visualize relationships across clusters
    sns.pairplot(features, hue='Cluster', palette='tab10', plot_kws={'alpha': 0.6})
    plt.suptitle("Pair Plot of Clusters (HDBSCAN)", y=1.02)
    plt.show()

    return client_clusters
