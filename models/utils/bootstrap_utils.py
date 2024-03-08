import numpy as np

def bootstrap_samples(scores_dict, n=5, simulate_variability=True, variability_std=0.05):
    """
    Generate bootstrap samples, introducing artificial variability if needed.
    
    Args:
    - scores_dict (dict): Dictionary mapping client IDs to single scores.
    - n (int): Number of bootstrap samples to generate.
    - simulate_variability (bool): Whether to introduce artificial variability.
    - variability_std (float): Standard deviation of the artificial variability to introduce.
    
    Returns:
    - dict: A dictionary where keys are client IDs and values are arrays representing simulated score distributions.
    """
    bootstrap_means = {}
    for client_id, score in scores_dict.items():
        if simulate_variability:
            # Simulate a distribution around the single score using normal distribution
            simulated_scores = np.random.normal(loc=score, scale=variability_std * score, size=n)
        else:
            # If not simulating variability, just replicate the single score
            simulated_scores = np.full(n, score)
        bootstrap_means[client_id] = simulated_scores
    return bootstrap_means
        

def simulate_score_distributions(hardware_scores, network_scores, data_quality_scores, n=5):
    """
    Simulate distributions for hardware, network, and data quality scores for each client.

    Args:
        - hardware_scores, network_scores, data_quality_scores (dict): Dictionaries mapping client IDs to scores.
        - n (int): Number of bootstrap samples to generate for each client per criterion.

    Returns:
        - dict: Nested dictionary containing bootstrap sample means for each criterion and client.
    """
    bootstrap_results = {
        'hardware': bootstrap_samples(hardware_scores, n),
        'network': bootstrap_samples(network_scores, n),
        'data_quality': bootstrap_samples(data_quality_scores, n)
    }
    return bootstrap_results

