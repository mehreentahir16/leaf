"""Script to run the baselines."""
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Use CPU only
import copy
import importlib
import numpy as np

import random
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt

import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server
from model import ServerModel

from utils.args import parse_args
from utils.model_utils import read_data
from client_selection_protocols import select_clients_randomly, select_clients_greedy, select_clients_price_based, client_selection_active, client_selection_pow_d, select_clients_resource_based, promethee_selection

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

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
        weights = np.array([0.1, 0.2, 0.7])
        selected_clients = promethee_selection(round_number, clients=clients, hardware_scores=hardware_scores, network_scores=network_scores, 
                                                data_quality_scores=data_quality_scores, weights=weights, costs=costs, num_clients=num_clients, budget=budget)
    else:
        raise ValueError("Invalid client selection strategy.")

    return selected_clients

def main():

    budget = None
    args = parse_args()
    selection_strategy = args.client_selection_strategy

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    tf.compat.v1.set_random_seed(123 + args.seed)

    model_path = '%s/%s.py' % (args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s' % (args.dataset, args.model)
    
    print('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    # Suppress tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.WARN)

    # Create 2 models
    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    # Create client model, and share params with server model
    tf.compat.v1.reset_default_graph()
    client_model = ClientModel(args.seed, *model_params)

    # Create server
    server = Server(client_model)

    # Create clients
    clients = setup_clients(args.dataset, client_model, args.use_val_set)
    # Initialize cumulative Shapley Values
    server.cumulative_shapley_values = {client.id: 0.0 for client in clients}
    client_ids, client_groups, client_num_samples, hardware_scores, network_scores, data_quality_scores, costs, losses = server.get_clients_info(clients)
    print('Clients in Total: %d' % len(clients))

    # Initial status
    print('--- Random Initialization ---')
    stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)
    print_stats(0, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)

    test_accuracies = []
    test_losses = []
    training_time = {}
    no_selected_clients = None

    # Define accuracy thresholds and initialize time tracking for each
    accuracy_thresholds = {50: None, 60: None, 70: None, 80: None, 90: None}
    cumulative_shapley_values_over_time = []
 
    total_training_time = 0
    total_unique_samples = 0
    unique_client_ids = set()

    # Simulate training
    for i in range(num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

        # if (i % 5 == 0) or i == 0:  # Select clients at the start and then every 5 rounds
        server.selected_clients = select_clients(selection_strategy, i, clients, client_num_samples, costs, hardware_scores, network_scores, data_quality_scores, losses, clients_per_round, budget=budget)

        c_ids, c_groups, c_num_samples, c_hardware_scores, c_network_scores, c_data_quality_scores, c_costs, c_losses = server.get_clients_info(server.selected_clients)

        print("===========Client info=============")
        print("selected client IDs", c_ids)

        # Simulate server model training on selected clients' data
        sys_metrics, download_time, estimated_training_time, upload_time = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch, simulate_delays=True)
        sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples)

        server.compute_shapley_values_tmc(active_clients=server.selected_clients, num_samples=50, epsilon=0.01)
        
        # Update server model
        server.update_model()

        simulated_total_time = download_time + estimated_training_time + upload_time 
        print("simulated total time", simulated_total_time)
        training_time[i] = simulated_total_time
        total_training_time += simulated_total_time

        # Test model
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            current_accuracy, current_loss = print_stats(i + 1, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)
            test_accuracies.append(current_accuracy)
            test_losses.append(current_loss)
            for threshold in accuracy_thresholds.keys():
                if current_accuracy >= threshold and accuracy_thresholds[threshold] is None:
                    accuracy_thresholds[threshold] = total_training_time 

        # Store cumulative Shapley Values after this round
        cumulative_shapley_values_over_time.append(copy.deepcopy(server.cumulative_shapley_values))

    print("Model training time: ", total_training_time)
    
    # Save server model
    ckpt_path = os.path.join('checkpoints', args.dataset)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server.save_model(os.path.join(ckpt_path, '{}.weights.h5'.format(args.model)))
    print('Model weights saved in path: %s' % save_path)

    # Saving results including training_time for each round
    results = {
        "accuracies": test_accuracies,
        "losses": test_losses,
        "training_time": total_training_time,
        "time_to_reach_accuracy_thresholds": accuracy_thresholds,
        "cumulative_shapley_values": cumulative_shapley_values_over_time
    }

    file_name = f"results/data_{args.dataset}_clients_{args.clients_per_round}_batch{args.batch_size}_results.pkl"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, "wb") as f:
        pickle.dump(results, f)

    print(f'Results saved to {file_name}')


def create_clients(users, groups, train_data, test_data, model):
    if len(groups) == 0:
        groups = [[] for _ in users]
    clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
    return clients


def setup_clients(dataset, model=None, use_val_set=False):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    eval_set = 'test' if not use_val_set else 'val'
    train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset, 'data', eval_set)

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    clients = create_clients(users, groups, train_data, test_data, model)

    return clients


def get_stat_writer_function(ids, groups, num_samples, args):

    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, partition, args.metrics_dir, '{}_{}'.format(args.metrics_name, 'stat'))

    return writer_fn


def get_sys_writer_function(args):

    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, 'train', args.metrics_dir, '{}_{}'.format(args.metrics_name, 'sys'))

    return writer_fn

def print_stats(num_round, server, clients, num_samples, args, writer, use_val_set):
    
    train_stat_metrics = server.test_model(clients, set_to_use='train')
    print_metrics(train_stat_metrics, num_samples, prefix='train_')
    writer(num_round, train_stat_metrics, 'train')

    eval_set = 'test' if not use_val_set else 'val'
    test_stat_metrics = server.test_model(clients, set_to_use=eval_set)
    print_metrics(test_stat_metrics, num_samples, prefix='{}_'.format(eval_set))
    writer(num_round, test_stat_metrics, eval_set)
    if eval_set == 'test':  
        test_accuracy, test_loss = server.calculate_test_accuracy(test_stat_metrics, num_samples)
        print("Calculated test_accuracy:", test_accuracy)
        print("calculated test_loss:", test_loss)
    return test_accuracy, test_loss


def print_metrics(metrics, weights, prefix=''):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights)]
    metric_names = metrics_writer.get_metrics_names(metrics)
    to_ret = None
    for metric in metric_names:
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        weighted_average = np.average(ordered_metric, weights=ordered_weights)
        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 weighted_average,
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))


if __name__ == '__main__':
    main()
