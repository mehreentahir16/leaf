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

import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server
from model import ServerModel
from client_selection import select_clients

from utils.args import parse_args
from utils.model_utils import read_data, get_label_flipping_config, add_gaussian_noise, rotate_data, shift_features, augment_data


from PCA import get_layerwise_updates, apply_pca
from autoencoder import LayerWiseAutoencoder

from sklearn.decomposition import PCA


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

    layer_updates = get_layerwise_updates(update_weights)
    print(f"Number of layers: {len(layer_updates)}")
    for idx, layer_data in enumerate(layer_updates):
        print(f"Shape of updates for layer {idx}: {layer_data.shape}")

    layer_scalers = []
    normalized_layer_updates = []

    # Scale the PCA-transformed output
    for idx, layer_data in enumerate(layer_updates):
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(layer_data)
        normalized_layer_updates.append(normalized_data)
        layer_scalers.append(scaler)
        # Save the scaler for this layer
        joblib.dump(scaler, f'scaler_layer_{idx}.pkl')

    variance_retained = 0.95  # Adjust as needed
    pca_models = []
    reduced_layer_updates = []
    updated_layer_dims = []

    for idx, layer_data in enumerate(normalized_layer_updates):
        reduced_data, n_components, pca_model = apply_pca(idx, layer_data, desired_variance=variance_retained)
        
        # Append results
        reduced_layer_updates.append(reduced_data)
        updated_layer_dims.append(n_components)
        pca_models.append(pca_model)
        
        # Free memory
        del layer_data, reduced_data, pca_model
        gc.collect()

    augmented_layer_updates = augment_data(reduced_layer_updates)
    # Set bottleneck dimensions dynamically
    compression_factor = 4
    bottleneck_dims = [max(1, dim // compression_factor) for dim in updated_layer_dims]

    learning_rate = 0.001
    num_epochs = 50
    batch_size = 32  # As you've set

    # original_data = pca_transformed_scaled

    # # Augment data using multiple strategies
    # augmented_data_gaussian = augment_with_gaussian_noise(original_data, noise_level=0.01, n_augmented=1)
    # augmented_data_scaling = augment_with_scaling(original_data, scale_range=(0.98, 1.02), n_augmented=1)
    # augmented_data_interpolation = augment_with_interpolation(original_data, n_augmented=len(original_data))
    # augmented_data_smote = augment_with_smote(original_data, n_neighbors=5, n_augmented=2)

    # # Combine all data
    # augmented_data = np.vstack((original_data))

    # print("augmented data length", len(augmented_data))

    # from sklearn.utils import shuffle

    # augmented_data = shuffle(augmented_data, random_state=42)

    autoencoder = LayerWiseAutoencoder(updated_layer_dims, bottleneck_dims, learning_rate=learning_rate)

    # Train Autoencoder
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        print("Starting Autoencoder Training...")
        autoencoder.train(sess, augmented_layer_updates, num_epochs=num_epochs, batch_size=batch_size)
        
        # Save the trained Autoencoder model
        saver = tf.train.Saver()
        saver.save(sess, 'autoencoder_model/autoencoder.ckpt')
        print("LayerWiseAutoencoder model saved successfully.")

        # Compute training errors per layer
        training_errors_per_layer = autoencoder.compute_layer_errors(sess, reduced_layer_updates)
        # Save mean and std of errors per layer for thresholding
        layer_error_means = []
        layer_error_stds = []
        for idx, layer_errors in enumerate(training_errors_per_layer):
            mean_error = np.mean(layer_errors)
            std_error = np.std(layer_errors)
            layer_error_means.append(mean_error)
            layer_error_stds.append(std_error)
            print(f"Layer {idx} error mean: {mean_error}, std: {std_error}")
        # avg_val_loss = cross_validate_autoencoder(
        #     X_data=pca_transformed_scaled,
        #     bottleneck_dim=3,
        #     learning_rate=0.001,
        #     num_epochs=100,
        #     batch_size=32,
        #     K=5
        # )

        # print("avg loss in validation", avg_val_loss)

        # anomaly_threshold = np.percentile(training_errors, 90) 
    
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
    total_unique_samples = 0
    unique_client_ids = set()

    # Load once outside the main training loop
    with tf.Session() as sess:
        saver.restore(sess, 'autoencoder_model/autoencoder.ckpt')
        print("Autoencoder model loaded successfully.")

        # Load scalers for each layer
        pca_models = []
        for idx in range(len(updated_layer_dims)):
            pca = joblib.load(f'memmaps/pca_layer_{idx}.pkl')
            pca_models.append(pca)

        # Simulate training
        for i in range(num_rounds):
            print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))

            server.selected_clients = select_clients(selection_strategy, i, clients, client_num_samples, costs, hardware_scores, network_scores, data_quality_scores, losses, clients_per_round, budget=budget)
            for client in server.selected_clients:
                if client.is_malicious:  # Condition when client becomes malicious
                    if args.attack == 'label_flip':
                        client.flip_labels(label_flip_config)
                        print(f"Malicious client {client.id} in Round {i} performing label flipping.")
                    # elif args.attack_type == 'random_noise':
                    #     client.add_noise(random_noise_config['noise_level'])
                    #     print(f"Malicious client {client.id} in Round {i} adding random noise.")

            c_ids, c_groups, c_num_samples, c_hardware_scores, c_network_scores, c_data_quality_scores, c_costs, c_losses, _, _, _ = server.get_clients_info(server.selected_clients)

            print("===========Client info=============")

            # no_selected_clients = (len(server.selected_clients))
            # print("no_selected_clients", no_selected_clients)
            # avg_num_samples = sum(c_num_samples.values())/no_selected_clients
            # print("avg_num_samples", avg_num_samples)
            # new_clients = {client_id: samples for client_id, samples in c_num_samples.items() if client_id not in unique_client_ids}
            # total_unique_samples += sum(new_clients.values())
            # unique_client_ids.update(new_clients.keys())  # Update the set of unique client IDs

            # print(f"Total unique training samples so far: {total_unique_samples}")

            # Simulate server model training on selected clients' data
            sys_metrics, gradient_magnitudes, gradient_variances, update_weights, download_time, estimated_training_time, upload_time = server.train_model(num_epochs=args.num_epochs, batch_size=args.batch_size, minibatch=args.minibatch, simulate_delays=True)
            sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples)

            # Flatten, scale, and transform updates
            layer_updates_inference = get_layerwise_updates(update_weights)
            # Normalize the updates per layer using the saved scalers
            normalized_layer_updates_inference = []
            for idx, layer_data in enumerate(layer_updates_inference):
                scaler = layer_scalers[idx]
                normalized_data = scaler.transform(layer_data)
                normalized_layer_updates_inference.append(normalized_data)

            # Apply PCA to each layer
            reduced_layer_updates_inference = []
            for idx, layer_data in enumerate(normalized_layer_updates_inference):
                pca = pca_models[idx]
                reduced_data = pca.transform(layer_data)
                reduced_layer_updates_inference.append(reduced_data)

            # Compute per-client reconstruction errors per layer
            reconstruction_errors_per_layer = autoencoder.compute_layer_errors(sess, reduced_layer_updates_inference)

            # Aggregate errors per client
            num_clients = len(server.selected_clients)
            total_reconstruction_errors = np.zeros(num_clients)
            for layer_errors in reconstruction_errors_per_layer:
                total_reconstruction_errors += layer_errors  # Sum errors across layers

            # Compute anomaly threshold based on training errors (e.g., 95th percentile)
            anomaly_threshold = np.percentile(total_reconstruction_errors, 85)

            # Identify anomalous clients
            flagged_clients = []
            for idx, error in enumerate(total_reconstruction_errors):
                if error > anomaly_threshold:
                    flagged_clients.append(server.selected_clients[idx].id)

            print("Total Reconstruction errors:", total_reconstruction_errors)
            print("Anomaly threshold:", anomaly_threshold)
            print("Anomalous clients flagged:", flagged_clients)

            # Determine layer-wise thresholds
            # layer_percentiles = [90] * len(reconstruction_errors_per_layer)  # Adjust as needed
            # layer_thresholds = []
            # for layer_idx, layer_errors in enumerate(reconstruction_errors_per_layer):
            #     threshold = np.percentile(layer_errors, layer_percentiles[layer_idx])
            #     layer_thresholds.append(threshold)
            #     print(f"Layer {layer_idx} anomaly threshold (percentile {layer_percentiles[layer_idx]}): {threshold}")

            # # Flag anomalous clients based on layer-wise thresholds
            # flagged_clients = set()
            # num_clients = len(server.selected_clients)
            # for layer_idx, layer_errors in enumerate(reconstruction_errors_per_layer):
            #     threshold = layer_thresholds[layer_idx]
            #     for client_idx in range(num_clients):
            #         if layer_errors[client_idx] > threshold:
            #             print(f"flagging client {server.selected_clients[client_idx].id}")
            #             flagged_clients.add(server.selected_clients[client_idx].id)

            # flagged_clients = list(flagged_clients)
            # print("Anomalous clients flagged (layer-wise):", flagged_clients)

            
            # # Identify and flag anomalies based on reconstruction errors
            # non_anomalous_clients = [
            #     client for idx, client in enumerate(server.selected_clients)
            #     if reconstruction_errors[idx] <= anomaly_threshold
            # ]
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

    print("Model training time: ", total_training_time)
    
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
        # "avg_number_of_samples": avg_num_samples,
        # "total_unique_training_samples": total_unique_samples
    }

    file_name = f"results/data_{args.dataset}_selection_{args.client_selection_strategy}_clients_{args.clients_per_round}_budget_{budget}_results_test.pkl"

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