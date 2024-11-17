import json
import joblib
import numpy as np
import os
from collections import defaultdict
import tensorflow as tf
from sklearn.preprocessing import StandardScaler 

from utils.args import label_flipping_config


def batch_data(data, batch_size, seed):
    '''
    data is a dict := {'x': [numpy array], 'y': [numpy array]} (on one client)
    returns x, y, which are both numpy array of length: batch_size
    '''
    data_x = data['x']
    data_y = data['y']

    # randomly shuffle data
    np.random.seed(seed)
    rng_state = np.random.get_state()
    np.random.shuffle(data_x)
    np.random.set_state(rng_state)
    np.random.shuffle(data_y)

    # loop through mini-batches
    for i in range(0, len(data_x), batch_size):
        batched_x = data_x[i:i+batch_size]
        batched_y = data_y[i:i+batch_size]
        yield (batched_x, batched_y)

def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir,f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])

        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        groups: list of group ids; empty list if none found
        train_data: dictionary of train data
        test_data: dictionary of test data
    '''
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data

def get_model_size(model):
    """
    Estimates the size of the TensorFlow model by summing the sizes of its trainable parameters.
    
    Args:
    - model: An instance of a TensorFlow model.
    
    Returns:
    - The estimated size of the model in bytes.
    """
    total_size = 0
    for variable in model.sess.run(tf.trainable_variables()):
        # Each variable is a tf.Tensor or tf.Variable, get its size in bytes
        total_size += variable.size * variable.itemsize
    return total_size

def get_update_size(update):
    """
    Calculate the size of the update in bytes.
    
    Args:
    - update: A list of numpy arrays representing the model weights after training.
    
    Returns:
    - total_size: The total size of the update in bytes.
    """
    total_size = sum(array.nbytes for array in update)
    return total_size

def get_label_flipping_config(dataset_name):
    config = label_flipping_config.get(dataset_name.lower())
    if config is None:
        raise ValueError(f"No label flipping configuration found for dataset: {dataset_name}")
    return config

def flip_labels(client_data):
    """
    Flip labels of '8' to '3' in the client's training data.

    Args:
        client_data (dict): A dictionary containing the training data with keys:
                            - 'x': List of input samples
                            - 'y': List of labels corresponding to input samples
    """
    # Modify labels in-place
    client_data['y'] = [3 if label == 8 else label for label in client_data['y']]

def add_gaussian_noise(data, mean=0, std=0.01):
    noise = np.random.normal(mean, std, data.shape)
    augmented_data = data + noise
    return augmented_data

# def apply_smote(data, sampling_strategy='auto', random_state=42):
#     smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
#     augmented_data, _ = smote.fit_resample(data, np.zeros(data.shape[0]))  # Dummy labels
#     return augmented_data

def rotate_data(data, angle=15):
    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta),  np.cos(theta)]])
    if data.shape[1] >= 2:
        data_rotated = data.copy()
        data_rotated[:, :2] = data[:, :2].dot(rotation_matrix)
        return data_rotated
    else:
        return data

def shift_features(data, shift_range=0.05):
    shifts = np.random.uniform(-shift_range, shift_range, data.shape)
    shifted_data = data + shifts
    return shifted_data 

def augment_data(reduced_layer_updates):
    """
    Applies augmentation techniques to PCA-reduced data.
    
    Parameters:
    - reduced_layer_updates (list of np.ndarray): PCA-reduced data for each layer.
    
    Returns:
    - augmented_layer_updates (list of np.ndarray): Original and augmented data.
    """
    augmented_layer_updates = []
    
    for layer_data in reduced_layer_updates:
        # Original data
        augmented_data = [layer_data]
        
        # Add Gaussian noise
        noisy_data = add_gaussian_noise(layer_data, mean=0, std=0.01)
        augmented_data.append(noisy_data)
        
        # Shift features
        shifted_data = shift_features(layer_data, shift_range=0.05)
        augmented_data.append(shifted_data)
        
        # Optionally apply rotation if applicable
        # rotated_data = rotate_data(layer_data, angle=15)
        # augmented_data.append(rotated_data)
        
        # Concatenate all augmented data
        augmented_layer = np.vstack(augmented_data)
        augmented_layer_updates.append(augmented_layer)
    
    return augmented_layer_updates

# Function to normalize updates for each layer
def normalize_layer_updates(layer_updates):
    """
    Normalize layer updates using StandardScaler and save the scalers for future use.
    
    Parameters:
    - layer_updates (list of np.ndarray): List where each element corresponds to a layer's updates.
    
    Returns:
    - normalized_layer_updates (list of np.ndarray): List of normalized updates for each layer.
    - scalers (list of StandardScaler): List of fitted scalers for each layer.
    """
    normalized_layer_updates = []
    scalers = []
    
    for idx, layer_data in enumerate(layer_updates):
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(layer_data)
        normalized_layer_updates.append(normalized_data)
        scalers.append(scaler)
        
        # Save the scaler for this layer for later use
        joblib.dump(scaler, f'scaler_layer_{idx}.pkl')
        print(f"Scaler for layer {idx} saved at 'scaler_layer_{idx}.pkl'")
    
    return normalized_layer_updates, scalers

# Function to normalize reduced updates before autoencoder
def normalize_reduced_updates(reduced_layer_updates, reduced_scalers_dir='reduced_scalers'):
    """
    Normalize reduced layer updates using StandardScaler and save the scalers for future use.
    
    Parameters:
    - reduced_layer_updates (list of np.ndarray): List where each element corresponds to a layer's reduced updates.
    - reduced_scalers_dir (str): Directory to store scalers for reduced data.
    
    Returns:
    - normalized_reduced_layer_updates (list of np.ndarray): List of normalized reduced updates for each layer.
    - reduced_scalers (list of StandardScaler): List of fitted scalers for each layer.
    """
    normalized_reduced_layer_updates = []
    reduced_scalers = []
    
    os.makedirs(reduced_scalers_dir, exist_ok=True)
    
    for idx, layer_data in enumerate(reduced_layer_updates):
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(layer_data)
        normalized_reduced_layer_updates.append(normalized_data)
        reduced_scalers.append(scaler)
        
        # Save the scaler for this layer
        joblib.dump(scaler, f'{reduced_scalers_dir}/reduced_scaler_layer_{idx}.pkl')
        print(f"Scaler for reduced layer {idx} saved at '{reduced_scalers_dir}/reduced_scaler_layer_{idx}.pkl'")
    
    return normalized_reduced_layer_updates
