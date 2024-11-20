import math
import copy
import random
import threading
import numpy as np
import concurrent.futures

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY
from utils.client_resource_utils import calculate_hardware_score, calculate_network_score, calculate_data_quality_score

class Server:
    
    def __init__(self, client_model):
        self.client_model = client_model
        self.model = client_model.get_params()
        self.selected_clients = []
        self.updates = []
        # Initialize cumulative Shapley Values for all clients
        self.cumulative_shapley_values = {}
        self.global_model_backup = copy.deepcopy(self.model)

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
            comp, num_samples, update, d_time, t_time, u_time = c.train(num_epochs, batch_size, minibatch, simulate_delays)
            with threading.Lock():
                sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
                sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
                sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
                self.updates.append((c.id, num_samples, update))
                # Store times for each client
                download_times.append(d_time)
                training_times.append(t_time)
                upload_times.append(u_time)

        # Use ThreadPoolExecutor to limit the number of concurrent threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(train_client, c) for c in clients]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # wait for all threads to complete

        # Use the maximum time spent in any operation across all clients as the simulated time for that operation
        total_download_time = max(download_times) if download_times else 0
        total_training_time = max(training_times) if training_times else 0
        total_upload_time = max(upload_times) if upload_times else 0

        return sys_metrics, total_download_time, total_training_time, total_upload_time
    
    def get_clients_by_ids(self, client_ids):
        """
        Retrieves client objects based on their IDs.
        
        Args:
            client_ids (list): List of client IDs.
        
        Returns:
            list: List of Client objects matching the provided IDs.
        """
        return [client for client in self.selected_clients if client.id in client_ids]
    
    def compute_shapley_values_tmc(self, active_clients, num_samples=100, epsilon=0.01):
        """
        Computes Shapley Values for active clients using Truncated Monte Carlo.
        
        Args:
            active_clients (list): List of Client objects participating in the current round.
            num_samples (int): Number of sampled permutations for approximation.
            epsilon (float): Threshold for truncation based on marginal contribution.
        
        Updates:
            self.cumulative_shapley_values: Accumulates the computed SVs for each active client.
        """
        client_ids = [client.id for client in active_clients]
        self.global_model_backup = copy.deepcopy(self.model)  # Backup before permutations

        # Initialize marginal contributions storage
        marginal_contributions = {client_id: [] for client_id in client_ids}

        # Initialize truncation depth tracking
        truncation_depths = []

        for _ in range(num_samples):
            # Sample a random permutation of active clients
            perm = random.sample(client_ids, len(client_ids))

            coalition = []
            V_prev = self.evaluate_model_permutation(coalition)

            truncation_point = 0  # To track how many clients were added before truncation

            for client_id in perm:
                coalition.append(client_id)
                V_new = self.evaluate_model_permutation(coalition)
                marginal = V_new - V_prev
                if marginal < epsilon:
                    truncation_point += 1
                    break  # Truncate the permutation
                marginal_contributions[client_id].append(marginal)
                V_prev = V_new
                truncation_point += 1

            truncation_depths.append(truncation_point)

        # Calculate mean and confidence intervals
        for client_id, contributions in marginal_contributions.items():
            if contributions:
                mean = sum(contributions) / len(contributions)
                variance = sum((x - mean) ** 2 for x in contributions) / len(contributions)
                std_dev = math.sqrt(variance)
                confidence_interval = (mean - 1.96 * std_dev / math.sqrt(len(contributions)),
                                       mean + 1.96 * std_dev / math.sqrt(len(contributions)))
                # Update cumulative SV
                self.cumulative_shapley_values[client_id] += mean
            else:
                self.cumulative_shapley_values[client_id] += 0.0

        # Analyze truncation depths
        average_truncation = sum(truncation_depths) / len(truncation_depths) if truncation_depths else 0
        print(f"Average truncation depth: {average_truncation}")

    def evaluate_model_permutation(self, coalition):
        """
        Evaluates the model performance for a given coalition of clients.

        Args:
            coalition (list): List of client IDs contributing to the model.

        Returns:
            performance_metric (float): The performance of the aggregated model (e.g., accuracy).
        """
        if not coalition:
            return 0.0  # Baseline performance without any client updates

        # Extract updates for the coalition
        updates_subset = [update for update in self.updates if update[0] in coalition]

        # Aggregate the subset of updates
        aggregated_model = self.aggregate_updates_fedavg_subset(updates_subset)

        # Backup the current global model
        original_model = copy.deepcopy(self.model)

        # Set the aggregated model as the current model
        self.model = aggregated_model

        # Evaluate the model on a representative validation set
        # Here, you might need to define what constitutes a representative set
        # For simplicity, we'll assume using all clients' eval data
        # Alternatively, define a separate validation dataset
        clients_test = self.get_clients_by_ids(coalition)
        performance_metrics = self.test_model(clients_to_test=clients_test, set_to_use='val')

        weights_dict = {c_id: next((u[1] for u in self.updates if u[0] == c_id), 0) for c_id in coalition}

        print(f"Weights for coalition {coalition}: {weights_dict}")

        performance, _ = self.calculate_test_accuracy(performance_metrics, weights_dict)

        # Restore the original global model
        self.model = original_model

        return performance

    def update_model(self, method='fedavg'):
        if method == 'fedavg':
            self.aggregate_updates_fedavg()
        # Update the backup after aggregating
        self.global_model_backup = copy.deepcopy(self.model)

    def aggregate_updates_fedavg(self):
        print("inside fedavg...")
        total_weight = 0.
        base = [np.zeros_like(v.numpy(), dtype=np.float32) for v in self.updates[0][2]]  # Assuming update is a list of tensors
        for (client_id, client_samples, client_model) in self.updates:
            total_weight += client_samples
            for i, v in enumerate(client_model):
                base[i] += (client_samples * v.numpy().astype(np.float32))
        # Compute the average
        averaged_soln = [v / total_weight for v in base]

        self.model = averaged_soln
        self.updates = []

    def aggregate_updates_fedavg_subset(self, updates_subset):
        """
        Aggregates a subset of client updates using Federated Averaging.

        Args:
            updates_subset (list): List of updates (numpy arrays) from a subset of clients.

        Returns:
            averaged_soln (list): The aggregated global model parameters.
        """
        if not updates_subset:
            return self.model  # Return current model if no updates

        total_weight = sum([update[1] for update in updates_subset])
        if total_weight == 0:
            return self.model

        # Initialize the aggregated model parameters
        base = [np.zeros_like(v.numpy(), dtype=np.float32) for v in updates_subset[0][2]]  # Assuming update is a list of tensors

        for (client_id, num_samples, client_model) in updates_subset:
            for i, v in enumerate(client_model):
                base[i] += (num_samples * v.numpy().astype(np.float32))

        # Compute the average
        averaged_soln = [v / total_weight for v in base]

        return averaged_soln
    

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
    
    def calculate_test_accuracy(self, metrics, weights):
        """Calculate the weighted average for test_accuracy."""
        ordered_weights = [weights[c] for c in sorted(weights)]
        ordered_accuracy_metric = [metrics[c]['accuracy'] for c in sorted(metrics)]
        ordered_loss_metric = [metrics[c]['loss'] for c in sorted(metrics)]
        test_accuracy = np.average(ordered_accuracy_metric, weights=ordered_weights)
        test_loss = np.average(ordered_loss_metric, weights=ordered_weights)
        return test_accuracy, test_loss

    def save_model(self, path):
        """Saves the server model on checkpoints/dataset/model.ckpt."""
        # Save server model
        self.client_model.set_params(self.model)
        self.client_model.model.save_weights(path)
        return path

