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

def train_ipca(layer_idx, layer_data, n_components, batch_size=40, pca_dir='pca_models'):
    print(f"\nInitializing IncrementalPCA for layer {layer_idx} with n_components={n_components}")

    os.makedirs(pca_dir, exist_ok=True)

    # Memory-map the layer data to handle large datasets
    layer_data_memmap = np.memmap(f'memmap_layer_{layer_idx}.dat', dtype='float32', mode='w+', shape=layer_data.shape)
    layer_data_memmap[:] = layer_data
    del layer_data  # Free up memory for the original data

    # Initialize IPCA
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)

    # Fit IPCA in batches
    num_samples = layer_data_memmap.shape[0]
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        if end > num_samples:
            break  # Skip last batch if fewer than n_components
        batch = layer_data_memmap[start:end]
        ipca.partial_fit(batch)
        print(f"Layer {layer_idx}: IncrementalPCA fitted with batch {start} to {end}")
        del batch  # Free memory after fitting each batch
        gc.collect()

    # Transform the data in batches
    reduced_data = []
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        if end > num_samples:
            break  # Skip last batch if fewer than n_components
        batch = layer_data_memmap[start:end]
        reduced_batch = ipca.transform(batch)
        reduced_data.append(reduced_batch)
        del batch, reduced_batch  # Free memory after transforming each batch
        gc.collect()

    # Concatenate all reduced data batches into a single array
    reduced_data = np.vstack(reduced_data)

    # Save the IPCA model
    ipca_path = os.path.join(pca_dir, f'ipca_layer_{layer_idx}.pkl')
    joblib.dump(ipca, ipca_path)
    print(f"Layer {layer_idx}: IncrementalPCA model trained and saved at {ipca_path}")

    # Clean up memory
    del layer_data_memmap
    gc.collect()

    return reduced_data, ipca

def determine_n_components(layer_idx, layer_data, desired_variance=0.95, memmap_dir='memmaps'):
    """
    Determines the number of PCA components for a given layer based on desired variance.

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