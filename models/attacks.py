import numpy as np

def inject_random_gradient(update, noise_level=0.2):
    """
    Inject random noise into the model updates.
    
    Args:
        update (list): Model updates as a list of numpy arrays.
        noise_level (float): Standard deviation of the Gaussian noise.
        
    Returns:
        list: Noisy model updates.
    """
    noisy_update = []
    for weights in update:
        noise = np.random.normal(loc=0.0, scale=noise_level, size=weights.shape).astype(weights.dtype)
        noisy_weights = weights + noise
        noisy_update.append(noisy_weights)
    return noisy_update

def randomize_weights(update):
    """
    Completely randomize the model weights.
    
    Args:
        update (list): Model updates as a list of numpy arrays.
        
    Returns:
        list: Randomized model updates.
    """
    randomized_update = []
    for weights in update:
        randomized_weights = np.random.uniform(low=-1, high=1, size=weights.shape).astype(weights.dtype)
        randomized_update.append(randomized_weights)
    return randomized_update

