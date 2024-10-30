import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def calculate_contamination_rate(cluster_data):
    # Calculate the contamination rate based on data spread within the cluster
    mean = np.mean(cluster_data, axis=0)
    std = np.std(cluster_data, axis=0)
    
    # Using coefficient of variation to set contamination, limited to a reasonable range
    contamination_rate = min(max((std / mean).mean(), 0.01), 0.3)  # [0.01, 0.3] range
    return contamination_rate

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
        
        # Train Isolation Forest model on each cluster’s clients
        isolation_forest = IsolationForest(contamination=0.01, random_state=42, n_estimators=150, max_samples="auto", max_features=0.7)
        isolation_forest.fit(scaled_data)
        
        isolation_models[cluster_id] = isolation_forest
    
    return isolation_models

# In the main function after training
def check_for_anomalies(selected_clients, client_clusters, gradient_magnitudes, gradient_variances, isolation_models):
    valid_clients = []
    for client_id in selected_clients:
        cluster_id = client_clusters[client_id]
        
        if cluster_id == -1:
            print(f"Client {client_id} flagged as an outlier during initial clustering and excluded from aggregation")
            continue
        
        # Use the appropriate Isolation Forest model for the client's cluster
        if cluster_id in isolation_models:
            isolation_forest = isolation_models[cluster_id]
            client_data = np.array([[gradient_magnitudes[client_id], gradient_variances[client_id]]])
            
            # Predict if the client is an outlier
            if isolation_forest.predict(client_data)[0] == 1:  # 1 indicates inlier, -1 indicates outlier
                valid_clients.append(client_id)
            else:
                print(f"Client {client_id} flagged as anomalous by Isolation Forest and excluded from aggregation")
        else:
            print(f"Cluster model not found for client {client_id}")
    
    return valid_clients
