import ot
import threading
import numpy as np
import tensorflow as tf
import concurrent.futures

from multiprocessing import Pool, cpu_count

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY
from utils.client_resource_utils import calculate_hardware_score, calculate_network_score, calculate_data_quality_score

class Server:
    
    def __init__(self, client_model):
        self.client_model = client_model
        self.model = client_model.get_params()
        self.selected_clients = []
        self.updates = []

    def train_model(self, num_epochs, batch_size=10, minibatch=None, clients=None, simulate_delays=True):
        if clients is None:
            clients = self.selected_clients

        print("number of clients server is calling train on...", len(clients))

        sys_metrics = {c.id: {BYTES_WRITTEN_KEY: 0, BYTES_READ_KEY: 0, LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
        
        # Lists to hold times for each client
        download_times = []
        training_times = []
        upload_times = []

        def train_client(c):
            self.updates = []
            c.model.set_params(self.model)
            comp, num_samples, update,  d_time, t_time, u_time = c.train(num_epochs, batch_size, minibatch, simulate_delays)
            with threading.Lock():
                sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
                sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
                sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
                self.updates.append((num_samples, update))
                # Store times for each client
                download_times.append(d_time)
                training_times.append(t_time)
                upload_times.append(u_time)

         # Use ThreadPoolExecutor to limit the number of concurrent threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(train_client, c) for c in clients]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # wait for all threads to complete

        # Use the maximum time spent in any operation across all clients as the simulated time for that operation
        total_download_time = max(download_times) if download_times else 0
        total_training_time = max(training_times) if training_times else 0
        total_upload_time = max(upload_times) if upload_times else 0

        return sys_metrics, total_download_time, total_training_time, total_upload_time

    def update_model(self, method='emd'):
        if method == 'fedavg':
            self.aggregate_updates_fedavg()
        elif method == 'emd':
            self.aggregate_updates_wasserstein_barycenter()

    def aggregate_updates_fedavg(self):
        print("inside fedavg...")
        total_weight = 0.
        base = [np.zeros_like(v.numpy(), dtype=np.float64) for v in self.updates[0][1]]
        for (client_samples, client_model) in self.updates:
            total_weight += client_samples
            for i, v in enumerate(client_model):
                base[i] += (client_samples * v.numpy().astype(np.float64))
        averaged_soln = [v / total_weight for v in base]

        self.model = averaged_soln
        self.updates = []

    def aggregate_updates_wasserstein_barycenter(self):
        print("inside wasserstein_barycenter...")
        client_models = [update[1] for update in self.updates]
        weights = np.array([update[0] for update in self.updates])
        weights = weights / weights.sum()  # Normalize weights

        print(f"client_models type: {type(client_models)}")
        print(f"client_models[0] type: {type(client_models[0])}")
        print(f"client_models[0][0] type: {type(client_models[0][0])}")
        print(f"client_models[0][0] shape: {client_models[0][0].shape}")

        # Compute the Wasserstein barycenter
        barycenter_model = compute_wasserstein_barycenter(client_models, weights)

        # Set the aggregated model parameters
        self.client_model.set_params(barycenter_model)
        self.updates = []
        

    def test_model(self, clients_to_test, set_to_use='test'):
        """Tests self.model on given clients.

        Tests model on self.selected_clients if clients_to_test=None.

        Args:
            clients_to_test: list of Client objects.
            set_to_use: dataset to test on. Should be in ['train', 'test'].
        """
        metrics = {}

        if clients_to_test is None:
            clients_to_test = self.selected_clients

        for client in clients_to_test:
            client.model.set_params(self.model)
            c_metrics = client.test(set_to_use)
            metrics[client.id] = c_metrics
        
        return metrics

    def get_clients_info(self, clients):
        """Returns the ids, hierarchies and num_samples for the given clients.

        Returns info about self.selected_clients if clients=None;

        Args:
            clients: list of Client objects.
        """
        if clients is None:
            clients = self.selected_clients

        ids = [c.id for c in clients]
        groups = {c.id: c.group for c in clients}
        num_samples = {c.id: c.num_samples for c in clients}
        hardware_scores = {}
        network_scores = {}
        data_quality_scores = {}
        raw_costs = {}
        losses = {}

        # Train and test each client to get accuracy
        sys_metrics, _, _, _= self.train_model(clients=clients, num_epochs=1, simulate_delays=False)  
        accuracy_metrics = self.test_model(clients_to_test=clients, set_to_use='test')  # Get test metrics

        for c in clients:
            hw_config = c.hardware_config
            net_config = c.network_config
            
            # Calculate scores using the utility functions
            hardware_scores[c.id] = calculate_hardware_score(
                cpu_count=hw_config['CPU Count'], cpu_cores=hw_config['Cores'], 
                cpu_frequency=hw_config['Frequency'], gpu_presence=hw_config['GPU'], 
                ram=hw_config['RAM'], storage=hw_config['Storage'])
            
            network_scores[c.id] = calculate_network_score(
                bandwidth=net_config['Bandwidth'], latency=net_config['Latency'])

            # Retrieve and store accuracy for each client
            local_accuracy = accuracy_metrics[c.id].get('accuracy', 0)
            client_loss = accuracy_metrics[c.id].get('loss', 0)
            losses[c.id] = client_loss 

            # Calculate and store data quality score for each client
            data_quality_scores[c.id] = calculate_data_quality_score(data_size=num_samples[c.id], loss=client_loss)

            # Cost calculation
            comp_cost = sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY]  # Computational cost based on FLOPs
            net_cost = (sys_metrics[c.id][BYTES_WRITTEN_KEY] + sys_metrics[c.id][BYTES_READ_KEY]) * network_scores[c.id]  # Adjusted for network score
            dq_cost = data_quality_scores[c.id]
            # data_samples = num_samples[c.id]  # Data quality cost

            raw_costs[c.id] = (comp_cost ) + (net_cost ) + (dq_cost)

        # Normalize costs based on the highest raw cost
        max_cost = max(raw_costs.values())
        costs = {c_id: (cost / max_cost) * 10 for c_id, cost in raw_costs.items()}  # Scale to 0-100

        # Return all gathered information, including new scores
        return ids, groups, num_samples, hardware_scores, network_scores, data_quality_scores, costs, losses

    def save_model(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_model.set_params(self.model)
        self.client_model.model.save_weights(path)
        return path

def compute_wasserstein_barycenter(client_models, weights, reg=1.0, numItermax=10):
    """
    Compute the Wasserstein Barycenter of client model parameters.
    
    Args:
        client_models: List of client model parameter arrays.
        weights: List of weights for each client model.
        reg: Regularization parameter for Sinkhorn distance.
        numItermax: Maximum number of iterations for the Sinkhorn algorithm.
    
    Returns:
        barycenter_model: Aggregated model parameters as the Wasserstein Barycenter.
    """
    barycenter_model = []

    for i in range(len(client_models[0])):
        # Flatten the parameters for each client model
        param_matrix = np.array([client_model[i].numpy().flatten() for client_model in client_models])

        # Debugging output
        print(f"param_matrix shape: {param_matrix.shape}")

        # Ensure param_matrix has the correct shape and is non-empty
        if param_matrix.ndim != 2 or param_matrix.shape[0] == 0 or param_matrix.shape[1] == 0:
            raise ValueError(f"Invalid param_matrix shape: {param_matrix.shape}")

        # Compute the cost matrix
        M = ot.dist(param_matrix.T, param_matrix.T, metric='euclidean')

        # Debugging output for cost matrix
        print(f"Cost matrix M shape: {M.shape}")

        # Normalize the weights
        normalized_weights = weights / weights.sum()

        # Debugging output for weights
        print(f"Normalized weights shape: {normalized_weights.shape}")

        # Transpose param_matrix to match the expected shape for A (dim, n_hists)
        param_matrix_T = param_matrix.T  # **Changed Line**
        print(f"param_matrix_T shape: {param_matrix_T.shape}")  # **Debugging output for param_matrix_T**

        # Compute the Wasserstein barycenter for the i-th parameter
        barycenter_i = ot.bregman.barycenter(param_matrix_T, M, reg=reg, weights=normalized_weights, numItermax=numItermax)  # **Changed Line**

        # Debugging output for barycenter_i
        print(f"Barycenter shape: {barycenter_i.shape}")

        # Reshape the barycenter back to its original shape and convert to TensorFlow tensor
        barycenter_model.append(tf.convert_to_tensor(barycenter_i.T.reshape(client_models[0][i].shape), dtype=tf.float32))

    return barycenter_model