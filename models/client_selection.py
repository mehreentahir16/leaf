import numpy as np
from client_selection_protocols import select_clients_randomly, select_clients_greedy, select_clients_price_based, client_selection_active, client_selection_pow_d, select_clients_resource_based, promethee_selection

def select_clients(strategy, round_number, clients, client_num_samples, costs, hardware_scores, network_scores, 
                   data_quality_scores, losses, num_clients, budget=None):
    """Selects clients based on the given strategy, now including client_num_samples.

    Args:
        strategy (str): The client selection strategy.
        round_number (int): The current training round.
        clients (list): The list of client objects.
        client_num_samples (dict): The dictionary mapping client ids to their number of samples.
        costs (dict): The dictionary mapping client ids to their costs.
        hardware_scores (dict): Client hardware scores.
        network_scores (dict): Client network scores.
        data_quality_scores (dict): Client data quality scores.
        losses (dict): Client losses.
        num_clients (int): Number of clients to select.
        budget (optional): The budget for client selection. Defaults to None.

    Returns:
        list: The list of selected client objects.
    """
    if strategy == 'random': 
        selected_clients = select_clients_randomly(round_number, clients, costs, num_clients=num_clients, budget=budget)
    elif strategy == 'greedy': 
        selected_clients = select_clients_greedy(clients, costs, num_clients=num_clients, budget=budget)
    elif strategy == 'price_based':
        selected_clients = select_clients_price_based(clients, costs, num_clients=num_clients, budget=budget)
    elif strategy == 'resource_based':
        selected_clients = select_clients_resource_based(clients, hardware_scores, network_scores, costs, num_clients=num_clients, budget=budget)
    elif strategy == 'active':
        selected_clients = client_selection_active(clients, losses, costs, alpha1=0.75, alpha2=0.01, alpha3=0.1, num_clients=num_clients, budget=budget)
    elif strategy == 'pow-d':
        selected_clients = client_selection_pow_d(clients, client_num_samples, losses, costs, d=50, num_clients=num_clients, budget=budget)
    elif strategy == 'promethee':
        weights = np.array([0.1, 0.1, 0.4, 0.4])
        selected_clients = promethee_selection(round_number, clients=clients, hardware_scores=hardware_scores, network_scores=network_scores, 
                                                data_quality_scores=data_quality_scores, weights=weights, costs=costs, num_clients=num_clients, budget=budget)
    else:
        raise ValueError("Invalid client selection strategy.")

    return selected_clients