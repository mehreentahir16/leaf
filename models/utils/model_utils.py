import json
import numpy as np
import os
from collections import defaultdict
import tensorflow as tf


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

def read_dir(data_dir, introduce_mislabeled=False, mislabel_rate=0.0, mislabel_clients_fraction=0.0):
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
        
        # Determine which clients to mislabel
        if introduce_mislabeled:
            print(f"Introducing mislabeled data...")
            num_clients = len(cdata['users'])
            num_mislabel_clients = int(mislabel_clients_fraction * num_clients)
            mislabel_clients = np.random.choice(cdata['users'], size=num_mislabel_clients, replace=False)

            for user in mislabel_clients:
                labels = cdata['user_data'][user]['y']
                num_samples = len(labels)
                num_mislabeled = int(mislabel_rate * num_samples)

                mislabel_indices = np.random.choice(range(num_samples), size=num_mislabeled, replace=False)
                for idx in mislabel_indices:
                    original_label = labels[idx]
                    # For FEMNIST (assuming 62 classes)
                    new_label = np.random.choice([l for l in range(62) if l != original_label])
                    # For CelebA (binary classification)
                    # new_label = (original_label + 1) % 2

                    labels[idx] = new_label

                cdata['user_data'][user]['y'] = labels

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
    train_clients, train_groups, train_data = read_dir(train_data_dir, introduce_mislabeled=False, mislabel_rate=0.2, mislabel_clients_fraction=0.3)
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
    for variable in model.trainable_variables:
        variable_size = tf.size(variable).numpy()  # Calculate the size of the variable
        dtype_size = tf.dtypes.as_dtype(variable.dtype).size  # Get the size of the data type in bytes
        total_size += variable_size * dtype_size
    return total_size

def get_update_size(update):
    """
    Calculate the size of the update in bytes.
    
    Args:
    - update: A list of numpy arrays representing the model weights after training.
    
    Returns:
    - total_size: The total size of the update in bytes.
    """
    total_size = sum(array.numpy().nbytes for array in update)
    return total_size
