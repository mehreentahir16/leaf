from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np

def flatten_model_updates(update_weights):
    flattened_updates = []
    for client_id, client_update in update_weights.items():
        try:
            # Flatten each layer's weights and concatenate them
            flattened = np.concatenate([w.flatten() for w in client_update])
            flattened_updates.append(flattened)
        except AttributeError as e:
            print(f"Error flattening update for client {client_id}: {e}")
    return np.array(flattened_updates)

def perform_incremental_pca(flattened_updates, n_components=100, batch_size=100):
    """
    Performs Incremental PCA on the flattened updates.
    
    Parameters:
    - flattened_updates (np.ndarray): The flattened client updates.
    - n_components (int): Number of principal components to keep.
    - batch_size (int): Number of samples per batch.
    
    Returns:
    - pca_transformed (np.ndarray): The PCA-transformed data.
    - pca (IncrementalPCA): The fitted Incremental PCA model.
    - scaler (StandardScaler): The fitted scaler.
    """
    # Step 1: Normalize the Data
    scaler = StandardScaler()
    normalized_updates = scaler.fit_transform(flattened_updates)
    joblib.dump(scaler, 'scaler.pkl')  # Save for future use

    # Step 2: Initialize Incremental PCA
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    
    # Step 3: Fit Incremental PCA on Data in Batches
    for i in range(0, normalized_updates.shape[0], batch_size):
        batch = normalized_updates[i:i+batch_size]
        ipca.partial_fit(batch)
        print(f"Processed batch {i//batch_size + 1}/{int(np.ceil(normalized_updates.shape[0]/batch_size))}")
    
    # Step 4: Transform the Entire Dataset
    pca_transformed = ipca.transform(normalized_updates)
    
    # Save the Incremental PCA model
    joblib.dump(ipca, 'pca.pkl')
    
    return pca_transformed, ipca, scaler