from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler
import joblib
import numpy as np
import os
import gc
from sklearn.decomposition import PCA

def get_layerwise_updates(update_weights):
    """
    Collects layer-wise updates from clients.
    
    Parameters:
    - update_weights (dict): Dictionary where keys are client IDs and values are lists of layer updates.
    
    Returns:
    - layer_updates (list): A list where each element corresponds to a layer, containing a numpy array of updates from all clients for that layer.
    """
    num_layers = len(next(iter(update_weights.values())))
    layer_updates = [[] for _ in range(num_layers)]
    
    for client_id, client_update in update_weights.items():
        for i, layer in enumerate(client_update):
            layer_updates[i].append(layer.flatten())
    # Convert each layer's list of updates into a numpy array
    layer_updates = [np.array(layer_list, dtype=np.float32) for layer_list in layer_updates]
    return layer_updates


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
    remaining_samples = None
    for i in range(0, normalized_updates.shape[0], batch_size):
        batch = normalized_updates[i:i+batch_size]
        if batch.shape[0] < n_components:
            # Combine the leftover samples with the previous batch if possible
            if remaining_samples is not None:
                batch = np.vstack((remaining_samples, batch))
                remaining_samples = None
            else:
                remaining_samples = batch
                continue  # Skip this incomplete batch for now
        ipca.partial_fit(batch)
        print(f"Processed batch {i//batch_size + 1}/{int(np.ceil(normalized_updates.shape[0]/batch_size))}")
    
    # Step 4: Transform the Entire Dataset
    pca_transformed = ipca.transform(normalized_updates)
    
    # Save the Incremental PCA model
    joblib.dump(ipca, 'pca.pkl')
    
    return pca_transformed, ipca, scaler

def apply_pca(layer_idx, layer_data, desired_variance=0.95, memmap_dir='memmaps'):
    """
    Applies regular PCA to a given layer's data, utilizing memory mapping for large datasets.

    Parameters:
    - layer_idx (int): Index of the layer.
    - layer_data (np.ndarray): Normalized data for the layer.
    - desired_variance (float): The cumulative variance ratio to retain.
    - memmap_dir (str): Directory to store memory-mapped files.

    Returns:
    - reduced_data (np.ndarray): Data transformed with PCA.
    - n_components (int): Number of components used.
    - pca_model (PCA): The fitted PCA model.
    """
    print(f"\nApplying PCA to layer {layer_idx} with original dimension {layer_data.shape[1]}")
    
    # Define memmap path
    memmap_path = os.path.join(memmap_dir, f'layer_{layer_idx}.dat')
    os.makedirs(memmap_dir, exist_ok=True)
    
    # Save layer data as memmap to disk
    layer_data_memmap = np.memmap(memmap_path, dtype='float32', mode='w+', shape=layer_data.shape)
    layer_data_memmap[:] = layer_data
    del layer_data_memmap  # Flush to disk

    # Load memmap for processing
    layer_data_memmap = np.memmap(memmap_path, dtype='float32', mode='r', shape=layer_data.shape)
    
    # Initialize PCA with randomized SVD solver
    pca = PCA(n_components=desired_variance, svd_solver='auto')
    
    # Fit PCA on memmap data
    reduced_data = pca.fit_transform(layer_data_memmap)
    
    # Save the PCA model
    joblib.dump(pca, os.path.join(memmap_dir, f'pca_layer_{layer_idx}.pkl'))
    
    # Get the number of components
    n_components = pca.n_components_
    print(f"Layer {layer_idx} reduced to dimension {n_components}, explained variance ratio: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # Clean up memmap
    del layer_data_memmap
    gc.collect()
    
    return reduced_data, n_components, pca