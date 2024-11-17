"""Script to run the baselines."""
import time
import joblib
import importlib
import numpy as np
import os
import sys
import random
import pickle
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from sklearn.preprocessing import StandardScaler 
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import gc
from sklearn.mixture import GaussianMixture

import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server
from model import ServerModel
from client_selection import select_clients
from attacks import inject_random_gradient, randomize_weights

from utils.args import parse_args
from utils.model_utils import read_data, get_label_flipping_config, augment_data, normalize_layer_updates, normalize_reduced_updates


from PCA import get_layerwise_updates, determine_n_components, train_ipca
from autoencoder import LayerWiseAutoencoder


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'



def main():

    budget = None
    args = parse_args()
    selection_strategy = args.client_selection_strategy

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    tf.set_random_seed(123 + args.seed)

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
    tf.logging.set_verbosity(tf.logging.WARN)

    # Create 2 models
    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    # Create client model, and share params with server model
    tf.reset_default_graph()
    client_model = ClientModel(args.seed, *model_params)

    # Create server
    server = Server(client_model)

    # Create clients
    clients = setup_clients(args.dataset, client_model, args.use_val_set)
    client_ids, client_groups, client_num_samples, hardware_scores, network_scores, data_quality_scores, costs, losses, gradient_magnitudes, gradient_variances, update_weights = server.get_clients_info(clients)

    # sampled_client_ids = random.sample(list(update_weights.keys()), min(100, len(update_weights)))
    # sampled_update_weights = {client_id: update_weights[client_id] for client_id in sampled_client_ids}

    # sampled_layer_updates = get_layerwise_updates(sampled_update_weights)
    # normalized_sampled_layer_updates, scalers = normalize_layer_updates(sampled_layer_updates)
    # print(f"Collected and normalized layer-wise updates for {len(normalized_sampled_layer_updates)} layers.")

    # # Determine n_components for each layer using sampled data
    # desired_variance = 0.95  # 95% variance retention
    # n_components_per_layer = []

    # for layer_idx, layer_data in enumerate(normalized_sampled_layer_updates):
    #     _, n_components, _ = determine_n_components(layer_idx, layer_data, desired_variance=desired_variance)
    #     n_components_per_layer.append(n_components)

    # print(f"Determined n_components for all layers: {n_components_per_layer}")

    # layer_updates = get_layerwise_updates(update_weights)
    # print(f"Number of layers: {len(layer_updates)}")
    # for idx, layer_data in enumerate(layer_updates):
    #     print(f"Shape of updates for layer {idx}: {layer_data.shape}")

    # normalized_layer_updates, scalers = normalize_layer_updates(layer_updates)

    # ipca_models = []
    # reduced_layer_updates = []
    # ipca_batch_size = max(n_components_per_layer) + 2

    # for layer_idx, (layer_data, n_components) in enumerate(zip(normalized_layer_updates, n_components_per_layer)):
    #     reduced_data, ipca_model = train_ipca(
    #         layer_idx=layer_idx,
    #         layer_data=layer_data,
    #         n_components=n_components,
    #         batch_size=ipca_batch_size,  # Align with your per-round client updates
    #         pca_dir='pca_models'
    #     )
    #     reduced_layer_updates.append(reduced_data)
    #     ipca_models.append(ipca_model)
        
    #     # Free memory
    #     del layer_data, reduced_data
    #     gc.collect()

    # augmented_layer_updates = augment_data(reduced_layer_updates)
    # # Set bottleneck dimensions dynamically
    # compression_factor = 4
    # bottleneck_dims = [max(1, dim // compression_factor) for dim in n_components_per_layer]
    # scaled_reduced_updates = normalize_reduced_updates(augmented_layer_updates)

    # learning_rate = 0.001
    # num_epochs = 50
    # batch_size = 8  # As you've set

    # autoencoder = LayerWiseAutoencoder(layer_dims=n_components_per_layer, bottleneck_dims=bottleneck_dims, learning_rate=learning_rate)

    # # Train Autoencoder
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
        
    #     print("Starting Autoencoder Training...")
    #     autoencoder.train(sess, scaled_reduced_updates, num_epochs=num_epochs, batch_size=batch_size)
        
    #     # Save the trained Autoencoder model
    #     saver = tf.train.Saver()
    #     saver.save(sess, 'autoencoder_model/autoencoder.ckpt')
    #     print("LayerWiseAutoencoder model saved successfully.")

    #     # Compute training errors per layer
    #     training_errors_per_layer = autoencoder.compute_layer_errors(sess, scaled_reduced_updates)

        # gmm_dir = 'gmm_models'
        # os.makedirs(gmm_dir, exist_ok=True)

        # gmm_models = []
        # for layer_idx, layer_errors in enumerate(training_errors_per_layer):
        #     # Initialize GMM with 2 components (adjust as needed)
        #     gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
        #     # Reshape errors for GMM input
        #     layer_errors_reshaped = layer_errors.reshape(-1, 1)
        #     # Fit GMM on reconstruction errors
        #     gmm.fit(layer_errors_reshaped)
        #     gmm_models.append(gmm)
        #     # Save the fitted GMM
        #     gmm_path = os.path.join(gmm_dir, f'gmm_layer_{layer_idx}.pkl')
        #     joblib.dump(gmm, gmm_path)
        #     print(f"GMM for layer {layer_idx} trained and saved at {gmm_path}")

        # Save mean and std of errors per layer for thresholding
        # layer_error_means = []
        # layer_error_stds = []
        # for idx, layer_errors in enumerate(training_errors_per_layer):
        #     mean_error = np.mean(layer_errors)
        #     std_error = np.std(layer_errors)
        #     layer_error_means.append(mean_error)
        #     layer_error_stds.append(std_error)
            # print(f"Layer {idx} error mean: {mean_error}, std: {std_error}")
    
    print('Clients in Total: %d' % len(clients))

    # Initial status
    print('--- Random Initialization ---')
    stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)
    print_stats(0, server, clients, client_num_samples, args, stat_writer_fn, args.use_val_set)


    label_flip_config = get_label_flipping_config(args.dataset)
    # Set malicious clients based on the specified fraction
    num_malicious_clients = int(args.malicious_fraction * len(clients))
    malicious_clients = random.sample(clients, num_malicious_clients)
    for client in malicious_clients:
        client.is_malicious = True  # Mark the client as malicious

    test_accuracies = []
    test_losses = []
    training_time = {}
    no_selected_clients = None

    # Define accuracy thresholds and initialize time tracking for each
    accuracy_thresholds = {50: None, 60: None, 70: None, 80: None, 90: None}

    total_training_time = 0
    total_selection_time = 0
    total_unique_samples = 0
    unique_client_ids = set()
    total_TP = 0
    total_FP = 0
    total_FN = 0
    total_malicious = 0
     # Track all malicious and flagged clients across all rounds
    all_actual_malicious_clients = set()

    # Load once outside the main training loop
    with tf.Session() as sess:
        # saver.restore(sess, 'autoencoder_model/autoencoder.ckpt')
        # print("Autoencoder model loaded successfully.")

        # # Load GMMs outside the training rounds
        # loaded_gmm_models = []
        # gmm_dir = 'gmm_models'
        # for layer_idx in range(len(n_components_per_layer)):
        #     gmm_path = os.path.join(gmm_dir, f'gmm_layer_{layer_idx}.pkl')
        #     if os.path.exists(gmm_path):
        #         gmm = joblib.load(gmm_path)
        #         loaded_gmm_models.append(gmm)
        #         print(f"GMM for layer {layer_idx} loaded from {gmm_path}")
        #     else:
        #         print(f"GMM for layer {layer_idx} not found at {gmm_path}.")
        #         # Handle the case where GMM is not found, possibly skip flagging for this layer
        #         loaded_gmm_models.append(None)

        # # Load IPCA models
        # loaded_ipca_models = []
        # for idx in range(len(n_components_per_layer)):
        #     ipca = joblib.load(f'pca_models/ipca_layer_{idx}.pkl')
        #     loaded_ipca_models.append(ipca)
        # print("IPCA models loaded successfully.")

        # # Load scalers for each layer
        # loaded_scalers = []
        # for idx in range(len(layer_updates)):
        #     scaler = joblib.load(f'scaler_layer_{idx}.pkl')
        #     loaded_scalers.append(scaler)
        # print("Scalers loaded successfully.")

        # Simulate training
        for i in range(num_rounds):
            print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

            selection_time_start = time.time()

            server.selected_clients = select_clients(selection_strategy, i, clients, client_num_samples, costs, hardware_scores, network_scores, data_quality_scores, losses, clients_per_round, budget=budget)

            selecting_time_end = time.time()

            round_selection_time = selecting_time_end - selection_time_start
            total_selection_time +=round_selection_time

            round_actual_malicious_clients = set()
            for client in server.selected_clients:
                if client.is_malicious:  # Condition when client becomes malicious
                    if args.attack == 'label_flip':
                        client.flip_labels(label_flip_config)
                        print(f"Malicious client {client.id} in Round {i} performing label flipping.")
                    round_actual_malicious_clients.add(client.id)

            all_actual_malicious_clients.update(round_actual_malicious_clients)

            c_ids, c_groups, c_num_samples, c_hardware_scores, c_network_scores, c_data_quality_scores, c_costs, c_losses, _, _, _ = server.get_clients_info(server.selected_clients)

            print("===========Client info=============")

            # no_selected_clients = (len(server.selected_clients))
            print("selected_clients", c_ids)
            # avg_num_samples = sum(c_num_samples.values())/no_selected_clients
            # print("avg_num_samples", avg_num_samples)
            # new_clients = {client_id: samples for client_id, samples in c_num_samples.items() if client_id not in unique_client_ids}
            # total_unique_samples += sum(new_clients.values())
            # unique_client_ids.update(new_clients.keys())  # Update the set of unique client IDs

            # print(f"Total unique training samples so far: {total_unique_samples}")

            # Simulate server model training on selected clients' data
            sys_metrics, gradient_magnitudes, gradient_variances, update_weights, updates_list, download_time, estimated_training_time, upload_time = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch, simulate_delays=True)
            sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples)

            # Inject Byzantine attacks by manipulating updates for malicious clients
            # for client_id in round_actual_malicious_clients:
            #     if args.attack == 'random_gradient':
            #         update_weights[client_id] = inject_random_gradient(update_weights[client_id], noise_level=0.1)
            #         print(f"Client {client_id} performed Random Gradient Injection.")
            #     elif args.attack == 'random_weights':
            #         update_weights[client_id] = randomize_weights(update_weights[client_id])
            #         print(f"Client {client_id} performed Random Weight Update.")

            for idx, (c_id, num_samples, update) in enumerate(updates_list):
                if c_id in round_actual_malicious_clients:
                    updates_list[idx] = (c_id, num_samples, update_weights[c_id])

            # Update server.updates with the manipulated updates
            server.updates = updates_list.copy()

            # Flatten, scale, and transform updates
            # layer_updates_inference = get_layerwise_updates(update_weights)
            # # Normalize each layer update for inference
            # normalized_layer_updates_inference = []
            # for idx, layer_data in enumerate(layer_updates_inference):
            #     scaler = joblib.load(f'scaler_layer_{idx}.pkl')
            #     normalized_data = scaler.transform(layer_data)
            #     normalized_layer_updates_inference.append(normalized_data)

            # # Apply IPCA to each layer
            # reduced_layer_updates_inference = []
            # for idx, layer_data in enumerate(normalized_layer_updates_inference):
            #     ipca = loaded_ipca_models[idx]
            #     reduced_data = ipca.transform(layer_data)
            #     reduced_layer_updates_inference.append(reduced_data)

            # scaled_reduced_updates_inference = normalize_reduced_updates(reduced_layer_updates_inference)

            # # Compute per-client reconstruction errors per layer
            # reconstruction_errors_per_layer = autoencoder.compute_layer_errors(sess, scaled_reduced_updates_inference)

            # client_layer_flags = {client_idx: 0 for client_idx in range(len(server.selected_clients))}
            # for layer_idx, layer_errors in enumerate(reconstruction_errors_per_layer):
            #     gmm = loaded_gmm_models[layer_idx]
            #     if gmm is None:
            #         print(f"GMM for layer {layer_idx} is not available. Skipping anomaly detection for this layer.")
            #         continue  # Skip this layer if GMM is not loaded

            #     # Reshape errors for GMM input
            #     layer_errors_reshaped = layer_errors.reshape(-1, 1)
            #     # Compute probabilities under GMM
            #     log_probs = gmm.score_samples(layer_errors_reshaped)  # Log probabilities
            #     probs = np.exp(log_probs)  # Convert to actual probabilities

            #     # Define a probability threshold (e.g., 5th percentile)
            #     prob_threshold = np.percentile(probs, 10)

            #     # Increment flag count for clients whose probabilities are below the threshold
            #     for client_idx, prob in enumerate(probs):
            #         if prob < prob_threshold:
            #             client_layer_flags[client_idx] += 1

            # # Flag clients who are flagged in at least 2 layers
            # round_flagged_clients = set()
            # for client_idx, flag_count in client_layer_flags.items():
            #     if flag_count >= 1:
            #         flagged_client_id = server.selected_clients[client_idx].id
            #         round_flagged_clients.add(flagged_client_id)
            #         print(f"Flagging client {flagged_client_id} in Round {i} (Flagged in {flag_count} layers)")

            # Aggregate errors per client
            # num_clients = len(server.selected_clients)
            # total_reconstruction_errors = np.zeros(num_clients)
            # for layer_errors in reconstruction_errors_per_layer:
            #     total_reconstruction_errors += layer_errors  # Sum errors across layers

            # # Compute anomaly threshold based on training errors (e.g., 95th percentile)
            # anomaly_threshold = np.percentile(total_reconstruction_errors, 95)

            # # Identify anomalous clients
            # round_flagged_clients = set()
            # for idx, error in enumerate(total_reconstruction_errors):
            #     if error > anomaly_threshold:
            #         round_flagged_clients.add(server.selected_clients[idx].id)

            # Alternatively, Determine layer-wise thresholds 
            # layer_percentiles = [75] * len(reconstruction_errors_per_layer)  # Adjust as needed
            # layer_thresholds = []
            # for layer_idx, layer_errors in enumerate(reconstruction_errors_per_layer):
            #     threshold = np.percentile(layer_errors, layer_percentiles[layer_idx])
            #     layer_thresholds.append(threshold)

            # # Flag anomalous clients based on layer-wise thresholds
            # round_flagged_clients = set()
            # num_clients = len(server.selected_clients)
            # client_layer_flags = {client_idx: 0 for client_idx in range(num_clients)}  # Track layer flags per client
            # for layer_idx, layer_errors in enumerate(reconstruction_errors_per_layer):
            #     threshold = layer_thresholds[layer_idx]
            #     for client_idx in range(num_clients):
            #         if layer_errors[client_idx] > threshold:
            #             print(f"flagging client {server.selected_clients[client_idx].id}")
            #             # client_layer_flags[client_idx] += 1
            #             round_flagged_clients.add(server.selected_clients[client_idx].id)

            # # round_flagged_clients = {server.selected_clients[client_idx].id for client_idx, flag_count in client_layer_flags.items() if flag_count >= 2}
            # print("Anomalous clients flagged (flagged in 2 or more layers):", round_flagged_clients)

            # # === Performance Metrics Calculation for this Round ===
            # TP = len(round_flagged_clients.intersection(round_actual_malicious_clients))
            # FP = len(round_flagged_clients - round_actual_malicious_clients)
            # FN = len(round_actual_malicious_clients - round_flagged_clients)
            # total_TP += TP
            # total_FP += FP
            # total_FN += FN
            # total_malicious += len(round_actual_malicious_clients)

            # for client in server.selected_clients:
            #     if client.id in round_flagged_clients:
            #         # Reduce trust score for flagged (malicious) clients
            #         client.update_trust_score(is_malicious_action=True)  # Adjust penalty as needed
            #     else:
            #         # Reward non-flagged (non-malicious) clients with a slight trust increase
            #         client.update_trust_score(is_malicious_action=False)

            # Update server model excluding flagged clients
            server.update_model(flagged_clients=None)

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

    print("Model training time: ", total_training_time)

    # === Overall Training Performance ===
    mean_precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0
    mean_recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0
    mean_f1 = 2 * mean_precision * mean_recall / (mean_precision + mean_recall) if (mean_precision + mean_recall) > 0 else 0
    mean_detection_rate = total_TP / total_malicious if total_malicious > 0 else 0

    print("=== Overall Anomaly Detection Performance ===")
    print(f"Total True Positives (TP): {total_TP}")
    print(f"Total False Positives (FP): {total_FP}")
    print(f"Total False Negatives (FN): {total_FN}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Mean F1-Score: {mean_f1:.4f}")
    print(f"Mean Detection Rate: {mean_detection_rate:.4f}")
    
    # Save server model
    ckpt_path = os.path.join('checkpoints', args.dataset)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server.save_model(os.path.join(ckpt_path, '{}.ckpt'.format(args.model)))
    print('Model saved in path: %s' % save_path)

    # Saving results including training_time for each round
    results = {
        "accuracies": test_accuracies,
        "losses": test_losses,
        "training_time": total_training_time,
        "time_to_reach_accuracy_thresholds": accuracy_thresholds,
        "number_of_clients": no_selected_clients,  
        "selection_time": total_selection_time,
        # "total_unique_training_samples": total_unique_samples
    }

    file_name = f"results/data_{args.dataset}_selection_{args.client_selection_strategy}_clients_{args.clients_per_round}_client_selection_time_pow_d.pkl"

    # Ensure the directory exists
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, "wb") as f:
        pickle.dump(results, f)

    print(f'Results saved to {file_name}')

    # Close models
    server.close_model()

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

def calculate_test_accuracy(metrics, weights):
    """Calculate the weighted average for test_accuracy."""
    ordered_weights = [weights[c] for c in sorted(weights)]
    ordered_accuracy_metric = [metrics[c]['accuracy'] for c in sorted(metrics)]
    ordered_loss_metric = [metrics[c]['loss'] for c in sorted(metrics)]
    test_accuracy = np.average(ordered_accuracy_metric, weights=ordered_weights)
    test_loss = np.average(ordered_loss_metric, weights=ordered_weights)
    return test_accuracy, test_loss


def print_stats(num_round, server, clients, num_samples, args, writer, use_val_set):
    
    train_stat_metrics = server.test_model(clients, set_to_use='train')
    print_metrics(train_stat_metrics, num_samples, prefix='train_')
    writer(num_round, train_stat_metrics, 'train')

    eval_set = 'test' if not use_val_set else 'val'
    test_stat_metrics = server.test_model(clients, set_to_use=eval_set)
    print_metrics(test_stat_metrics, num_samples, prefix='{}_'.format(eval_set))
    writer(num_round, test_stat_metrics, eval_set)
    if eval_set == 'test':  # Ensure this matches how you determine the test set
        test_accuracy, test_loss = calculate_test_accuracy(test_stat_metrics, num_samples)
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