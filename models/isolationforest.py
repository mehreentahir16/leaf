import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy.stats import kstest, norm
from eif import iForest

def calculate_contamination_rate(cluster_data):
    epsilon=1e-6
    # Calculate the contamination rate based on data spread within the cluster
    mean = np.mean(cluster_data, axis=0)
    std = np.std(cluster_data, axis=0)

    # Replace zero means with epsilon to avoid division by zero
    mean = np.where(mean == 0, epsilon, mean)
    
    # Using coefficient of variation to set contamination, limited to a reasonable range
    contamination_rate = min(max((std / mean).mean(), 0.01), 0.3)  # [0.01, 0.1] range
    return contamination_rate
    # epsilon = 1e-6
    # mean = np.mean(cluster_data, axis=0)
    # std = np.std(cluster_data, axis=0)

    # # Replace zero means with epsilon to avoid division by zero
    # mean = np.where(mean == 0, epsilon, mean)

    # # Check for normality using the Kolmogorov-Smirnov test
    # ks_statistic, p_value = kstest(cluster_data, 'norm')

    # # Set contamination rate based on K-S p-value
    # if p_value > 0.05:  # Assume normal distribution if p > 0.05
    #     contamination_rate = min(max((std / mean).mean(), 0.05), 0.15)  # More conservative rate
    # else:
    #     # Apply Chebyshev's inequality-based contamination if not normal
    #     chebyshev_rate = (1 / ((2 ** 2)))  # Within 2 std deviations for a non-normal distribution
    #     contamination_rate = min(max(chebyshev_rate, 0.05), 0.3)

    # return contamination_rate


def apply_isolation_forest_scoring(client_clusters, gradient_magnitudes, gradient_variances):
    isolation_models = {}
    
    # Train an Isolation Forest model for each cluster on gradient magnitudes and variances
    for cluster_id in np.unique(list(client_clusters.values())):
        if cluster_id == -1:  # Skip outliers
            continue
        cluster_clients = [c_id for c_id, c in client_clusters.items() if c == cluster_id]
        
        # Prepare data for Isolation Forest
        cluster_data = np.array([[gradient_magnitudes[c_id], gradient_variances[c_id]] for c_id in cluster_clients])

        # Standardize features before training
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(cluster_data)

        print("scaled data:", scaled_data)

        contamination_rate = calculate_contamination_rate(scaled_data)
        
        # Train Isolation Forest model on each cluster’s clients
        isolation_forest = IsolationForest(contamination=contamination_rate, random_state=42, n_estimators=150, max_samples='auto', max_features=0.7)
        isolation_forest.fit(scaled_data)
        
        isolation_models[cluster_id] = isolation_forest
    
    return isolation_models

# In the main function after training
def check_for_anomalies(selected_clients, client_clusters, gradient_magnitudes, gradient_variances, isolation_models):
    valid_clients = []
    mal_clients = []
    for client_id in selected_clients:
        cluster_id = client_clusters[client_id]
        
        if cluster_id == -1:
            print(f"Client {client_id} flagged as an outlier during initial clustering")
            continue
        
        # Use the appropriate Isolation Forest model for the client's cluster
        if cluster_id in isolation_models:
            isolation_forest = isolation_models[cluster_id]
            client_data = np.array([[gradient_magnitudes[client_id], gradient_variances[client_id]]])
            
            # Predict if the client is an outlier
            if isolation_forest.predict(client_data)[0] == 1:  # 1 indicates inlier, -1 indicates outlier
                valid_clients.append(client_id)
            else:
                mal_clients.append(client_id)
                print(f"Client {client_id} flagged as anomalous by Isolation Forest")
        else:
            print(f"Cluster model not found for client {client_id}")
    
    return valid_clients
