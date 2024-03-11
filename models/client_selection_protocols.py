import random
import numpy as np
from copy import deepcopy
from scipy.stats import mannwhitneyu

def select_clients_randomly(my_round, possible_clients, num_clients):
    """Selects num_clients clients randomly from possible_clients.
    
    Note that within function, num_clients is set to
        min(num_clients, len(possible_clients)).

    Args:
        possible_clients: Clients from which the server can select.
        num_clients: Number of clients to select; default 20
    Return:
        list of (num_train_samples, num_test_samples)
    """
    np.random.seed(my_round)
    selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

    return selected_clients

def select_clients_greedy(possible_clients, costs, num_clients):
    """
    Selects clients based on the ratio of number of training samples to cost, preferring clients with more samples per cost unit.

    Args:
        possible_clients: List of Client objects from which to select.
        costs: Dictionary mapping client IDs to their associated costs.
        num_clients: Number of clients to select; default is 20.

    Returns:
        A list of selected Client objects.
    """
    # Sort possible clients based on the ratio of their training samples to their cost
    sorted_clients = sorted(possible_clients, key=lambda client: client.num_train_samples / costs[str(client.id)], reverse=True)
    
    # Select the top clients based on the calculated ratio
    return sorted_clients[:num_clients]

def select_clients_price_based(possible_clients, costs, num_clients):
    """
    Selects clients based on the lowest cost.

    Args:
        possible_clients: List of Client objects from which to select.
        costs: Dictionary mapping client IDs to their associated costs.
        num_clients: Number of clients to select; default is 20.

    Returns:
        A list of selected Client objects.
    """
    # Sort clients by their associated cost in ascending order
    sorted_clients = sorted(possible_clients, key=lambda client: costs[str(client.id)])
    
    # Select the clients with the lowest costs
    return sorted_clients[:num_clients]

def select_clients_resource_based(possible_clients, hardware_scores, num_clients):
    """
    Selects clients based on the lowest cost.

    Args:
        possible_clients: List of Client objects from which to select.
        costs: Dictionary mapping client IDs to their associated costs.
        num_clients: Number of clients to select; default is 20.

    Returns:
        A list of selected Client objects.
    """
    # Sort clients by their associated cost in ascending order
    sorted_clients = sorted(possible_clients, key=lambda client: hardware_scores[str(client.id)])
    
    # Select the clients with the lowest costs
    return sorted_clients[:num_clients]

def client_selection_active(clients, losses, alpha1=0.75, alpha2=0.01, alpha3=0.1, num_clients=20):
    """
    Active client selection based on performance (loss).

    Args:
        clients: List of Client objects.
        losses: Dictionary of losses for each client.
        alpha1: Proportion of clients to initially consider based on performance.
        alpha2: Weight for emphasizing loss in the selection process.
        alpha3: Proportion of clients to select randomly for diversity.
        num_clients: Total number of clients to select for the round.

    Returns:
        List of selected Client objects.
    """
    # Calculate values based on loss, emphasizing according to alpha2
    values = np.exp(np.array([losses[client.id] for client in clients]) * alpha2)
    
    # Drop a portion of clients based on alpha1, focusing on the higher losses
    num_drop = len(clients) - int(alpha1 * len(clients))
    drop_client_idxs = np.argsort([losses[client.id] for client in clients])[:num_drop]
    
    # Adjust probabilities for remaining clients
    probs = deepcopy(values)
    probs[drop_client_idxs] = 0  # Zero out dropped clients
    probs /= np.sum(probs)  # Normalize
    
    # Select clients based on adjusted probabilities
    num_select = int((1 - alpha3) * num_clients)
    selected_idxs = np.random.choice(range(len(clients)), num_select, p=probs, replace=False)
    
    # Select additional clients randomly for diversity
    not_selected = list(set(range(len(clients))) - set(selected_idxs))
    if alpha3 * num_clients > 1:  # Ensure there's a need for random selection
        selected_idxs_random = np.random.choice(not_selected, num_clients - num_select, replace=False)
        selected_client_idxs = np.concatenate((selected_idxs, selected_idxs_random), axis=0)
    else:
        selected_client_idxs = selected_idxs
    
    # Convert indices to client objects
    selected_clients = [clients[idx] for idx in selected_client_idxs]
    
    return selected_clients

def client_selection_pow_d(clients, client_num_samples, losses, d, num_clients):
    """
    Updated Power-of-Choice client selection 
    
    Args:
        clients: List of Client objects.
        client_num_samples: Dictionary mapping client IDs to their number of samples.
        losses: Dictionary mapping client IDs to their loss.
        d: Number of candidate clients to sample for potential selection.
        num_clients: Total number of clients to select for the round.
    
    Returns:
        List of selected client IDs based on the Power-of-Choice strategy.
    """
    # Extract IDs for all clients
    client_ids = [client.id for client in clients]

    # Calculate weights for each client based on their number of samples
    total_samples = sum(client_num_samples.values())
    weights = np.array([client_num_samples[id] / total_samples for id in client_ids])

    # Ensure probabilities sum to 1
    weights /= weights.sum()

    # Sample d candidate clients based on their weights
    candidate_indices = np.random.choice(range(len(clients)), size=d, p=weights, replace=False)
    candidate_clients = [clients[i] for i in candidate_indices]

    # Select clients with the highest loss from candidates
    candidate_clients.sort(key=lambda client: losses[client.id], reverse=True)
    selected_clients = candidate_clients[:num_clients]

    return selected_clients

# def stochastic_preference(distribution_a, distribution_b, alternative='greater'):
#     """
#     Compute preference score using the Mann-Whitney U test.

#     Parameters:
#     - distribution_a (np.array): Bootstrap distribution of scores for client A.
#     - distribution_b (np.array): Bootstrap distribution of scores for client B.
#     - alternative (str): Defines the alternative hypothesis ('greater', 'less', 'two-sided').

#     Returns:
#     - float: Preference score (0 to 1), indicating the strength of preference of A over B.
#     """

#     _, p_value = mannwhitneyu(distribution_a, distribution_b, alternative='two-sided')
#     # print(f"P-value: {p_value}")
#     preference_score = 1 - p_value  # Higher score indicates stronger preference
#     return preference_score

# def stochastic_promethee_selection(clients, bootstrap_results, weights, num_clients=20):
#     """
#     Selects clients based on Stochastic PROMETHEE method incorporating bootstrap distributions.

#     Parameters:
#     - clients (list of Client objects): All available clients.
#     - bootstrap_results (dict): Distributions of scores for each criterion per client.
#     - weights (dict): Weights for criteria (hardware, network, data_quality).
#     - num_clients (int): Number of clients to select.

#     Returns:
#     - list: Selected Client objects.
#     """
#     num_clients_total = len(clients)
#     aggregated_preferences = np.zeros((num_clients_total, num_clients_total))

#     for criterion, weight in weights.items():
#         # print(f"Processing criterion: {criterion}")
#         for i in range(num_clients_total):
#             for j in range(num_clients_total):
#                 if i != j:
#                     client_i_id = str(clients[i].id)  # Ensure client ID is used as a string if necessary
#                     client_j_id = str(clients[j].id)
#                     if client_i_id in bootstrap_results[criterion] and client_j_id in bootstrap_results[criterion]:
#                         distribution_i = bootstrap_results[criterion][client_i_id]
#                         distribution_j = bootstrap_results[criterion][client_j_id]
#                         # Ensure distributions are numpy arrays
#                         distribution_i = np.array(distribution_i)
#                         distribution_j = np.array(distribution_j)
#                         # print(f"Client {client_i_id} vs Client {client_j_id}: shapes {distribution_i.shape}, {distribution_j.shape}")
#                         preference_score = stochastic_preference(distribution_i, distribution_j)
#                         aggregated_preferences[i, j] += weight * preference_score
#                     else:
#                         break

#     phi_plus = np.sum(aggregated_preferences, axis=1)
#     phi_minus = np.sum(aggregated_preferences, axis=0)
#     net_flows = phi_plus - phi_minus

#     ranked_indices = np.argsort(-net_flows)[:num_clients]
#     selected_clients = [clients[idx] for idx in ranked_indices]

#     return selected_clients


def promethee_selection(clients, hardware_scores, network_scores, data_quality_scores, weights, num_clients, top_percentage=10):

    # Extract client IDs directly from the clients list
    client_ids = [client.id for client in clients]
    
    # Use these client IDs to align and extract scores
    hardware_scores = [hardware_scores[client_id] for client_id in client_ids]
    network_scores = [network_scores[client_id] for client_id in client_ids]
    data_quality_scores = [data_quality_scores[client_id] for client_id in client_ids]

    # Convert the aligned scores into a 2D numpy array (n_clients, n_criteria)
    X = np.array([hardware_scores, network_scores, data_quality_scores]).T
    
    # Step 2: Define preference functions (simplest is a linear preference function)
    def preference_function(a, b, q=0, p=1):
        diff = a - b
        if diff <= q:
            return 0
        elif diff > p:
            return 1
        else:
            return (diff - q) / (p - q)
    
    # Step 3: Calculate the pairwise preference matrix
    n_clients, n_criteria = X.shape
    F = np.zeros((n_clients, n_clients))
    
    for i in range(n_clients):
        for j in range(n_clients):
            if i != j:
                for k in range(n_criteria):
                    F[i, j] += weights[k] * preference_function(X[i, k], X[j, k])
                    
    # Step 4: Calculate the leaving and entering flows
    phi_plus = np.sum(F, axis=1) / (n_clients - 1)
    phi_minus = np.sum(F, axis=0) / (n_clients - 1)
    
    # Step 5: Calculate the net flows
    phi = phi_plus - phi_minus
    
    # Step 6: Rank the clients based on net flows
    ranking = sorted(list(enumerate(phi)), key=lambda x: x[1], reverse=True)

    # Determine the number of top clients based on the given percentage
    top_clients_count = int(np.ceil(len(clients) * (top_percentage / 100.0)))
    
    # Filter the top percentage of clients
    top_clients = ranking[:top_clients_count]
    
    # Randomly select 'num_clients' from the top percentage of clients
    selected_indices = random.sample([idx for idx, _ in top_clients], num_clients)

    if top_clients_count >= num_clients:
        # Return the selected clients
        return [clients[idx] for idx in selected_indices]
    else:
        return [clients[idx] for idx, _ in ranking[:num_clients]]
    
    # selected_clients = []
    # total_cost = 0
    # for idx, _ in ranking:
    #     client = clients[idx]
    #     cost = client.calculate_cost()
    #     if total_cost + cost <= budget:
    #         selected_clients.append(client)
    #         total_cost += cost
    #     else:
    #         pass

    # return selected_clients