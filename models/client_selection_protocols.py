import random
import numpy as np
from copy import deepcopy

def select_clients_randomly(my_round, possible_clients, costs, num_clients, budget):
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
    
    if budget is None:
        selected_clients = np.random.choice(possible_clients, num_clients, replace=False)
    else:
        selected_clients = []
        np.random.shuffle(possible_clients)
        total_cost = 0
        for client in possible_clients:
            client_cost = costs[str(client.id)]
            if total_cost + client_cost <= budget:
                selected_clients.append(client)
                total_cost += client_cost
            else:
                continue
    
    return selected_clients

def select_clients_greedy(possible_clients, costs, num_clients, budget):
    """
    Selects clients based on the ratio of number of training samples to cost, preferring clients with more samples per cost unit.

    Args:
        possible_clients: List of Client objects from which to select.
        costs: Dictionary mapping client IDs to their associated costs.
        num_clients: Number of clients to select; default is 20.

    Returns:
        A list of selected Client objects.
    """
    sorted_clients = sorted(possible_clients, key=lambda client: client.num_train_samples/costs[str(client.id)], reverse=True)
    
    if budget is None:
        return sorted_clients[:num_clients]
    
    selected_clients = []
    total_cost = 0
    for client in sorted_clients:
        client_cost = costs[str(client.id)]
        if total_cost + client_cost <= budget:
            selected_clients.append(client)
            total_cost += client_cost
        else:
            continue
    
    return selected_clients

def select_clients_price_based(possible_clients, costs, num_clients, budget=None):
    """
    Selects clients based on the lowest cost.

    Args:
        possible_clients: List of Client objects from which to select.
        costs: Dictionary mapping client IDs to their associated costs.
        num_clients: Number of clients to select; default is 20.

    Returns:
        A list of selected Client objects.
    """
    sorted_clients = sorted(possible_clients, key=lambda client: costs[str(client.id)])
    
    if budget is None:
        return sorted_clients[:num_clients]
    
    selected_clients = []
    total_cost = 0
    for client in sorted_clients:
        client_cost = costs[str(client.id)]
        if total_cost + client_cost <= budget:
            selected_clients.append(client)
            total_cost += client_cost
        else:
            continue
    
    return selected_clients

def select_clients_resource_based(possible_clients, hardware_scores, network_scores, costs, num_clients, budget=None):
    """
    Selects clients based on a combined score of hardware and network performance, aiming to minimize delay.

    Args:
        possible_clients: List of Client objects from which to select.
        hardware_scores: Dictionary mapping client IDs to their hardware scores.
        network_scores: Dictionary mapping client IDs to their network scores.
        num_clients: Number of clients to select.

    Returns:
        A list of selected Client objects.
    """
    combined_scores = {client: hardware_scores[str(client.id)] + network_scores[str(client.id)] for client in possible_clients}
    sorted_clients = sorted(possible_clients, key=lambda client: combined_scores[client], reverse=True)
    
    if budget is None:
        return sorted_clients[:num_clients]
    
    else:
        selected_clients = []
        total_cost = 0
        top_clients_count = int(np.ceil(len(possible_clients) * (20 / 100.0)))
        top_clients = sorted_clients[:top_clients_count]
        selected_indices_ = random.sample([idx for idx in top_clients], num_clients)
        for client in selected_indices_:
            client_cost = costs[str(client.id)]
            if total_cost + client_cost <= budget:
                selected_clients.append(client)
                total_cost += client_cost
            else: 
                continue
    
    return selected_clients


def client_selection_active(clients, losses, costs, alpha1=0.65, alpha2=0.05, alpha3=0.1, num_clients=20, budget=None):
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
    num_drop = len(clients) - int(alpha1 * len(clients))
    drop_client_idxs = np.argsort([losses[client.id] for client in clients])[:num_drop]
    
    probs = deepcopy(values)
    probs[drop_client_idxs] = 0
    probs /= np.sum(probs)
    
    selected_clients = []
    if budget is None:
        num_select = int((1 - alpha3) * num_clients)
        selected_idxs = np.random.choice(range(len(clients)), num_select, p=probs, replace=False)
        selected_clients = [clients[idx] for idx in selected_idxs]
    else:
        valid_indices = np.random.choice(range(len(clients)), 20, p=probs, replace=False)
        total_cost = 0
        for idx in valid_indices:
            if probs[idx] > 0:  # Client was not dropped
                client = clients[idx]
                client_cost = costs[client.id]
                if total_cost + client_cost <= budget:
                    selected_clients.append(client)
                    total_cost += client_cost
                else:
                    continue
    
    return selected_clients

def client_selection_pow_d(clients, client_num_samples, losses, costs, d, num_clients, budget=None):
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
    if budget is None:
        selected_clients = candidate_clients[:num_clients]
    else:
        selected_clients = []
        total_cost = 0
        for client in candidate_clients:
            client_cost = costs[str(client.id)]
            if total_cost + client_cost <= budget:
                selected_clients.append(client)
                total_cost += client_cost
            else:
                continue

    return selected_clients

def normalize_scores(scores):
    """Normalize an array of scores to a [0, 1] range."""
    min_score = np.min(scores)
    max_score = np.max(scores)
    return (scores - min_score) / (max_score - min_score)

def promethee_selection(my_round, clients, hardware_scores, network_scores, data_quality_scores, weights, costs, num_clients, budget=None, top_percentage=50):

    np.random.seed(my_round)

    # Extract client IDs directly from the clients list
    client_ids = [client.id for client in clients]
    trust_scores = [client.trust_score for client in clients]
    
    # Use these client IDs to align and extract scores
    hardware_scores = [hardware_scores[client_id] for client_id in client_ids]
    network_scores = [network_scores[client_id] for client_id in client_ids]
    data_quality_scores = [data_quality_scores[client_id] for client_id in client_ids]

    # Normalize the scores to a [0, 1] range
    hardware_scores = normalize_scores(np.array(hardware_scores))
    network_scores = normalize_scores(np.array(network_scores))
    data_quality_scores = normalize_scores(np.array(data_quality_scores))

    # Convert the aligned and normalized scores into a 2D numpy array (n_clients, n_criteria)
    X = np.array([hardware_scores, network_scores, data_quality_scores, trust_scores]).T
    
    # Step 2: Define preference functions (simplest is a linear preference function)
    def preference_function(a, b, q=0.05, p=1):
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

    top_clients_count = int(np.ceil(len(clients) * (top_percentage / 100.0)))
        
    # Filter the top percentage of clients
    top_clients = ranking[:top_clients_count]
    
    # Randomly select 'num_clients' from the top percentage of clients
    selected_indices_ = random.sample([idx for idx, _ in top_clients], num_clients)

    if budget is None:
        # Determine the number of top clients based on the given percentage
        return [clients[idx] for idx in selected_indices_]
        
        # return [clients[idx] for idx, _ in ranking[:num_clients]]

    else:
        selected_indices = []
        total_cost = 0
        # selected_indices_ = np.random.shuffle(selected_indices_)
        for idx in selected_indices_:
            client_cost = costs[str(clients[idx].id)]
            if total_cost + client_cost <= budget:
                selected_indices.append(idx)
                total_cost += client_cost   
            else:
                continue 
        print("total cost", total_cost)

    # Return the selected clients
    return [clients[idx] for idx in selected_indices]
