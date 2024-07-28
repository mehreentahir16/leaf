import ot
import threading
import numpy as np
import tensorflow as tf
import concurrent.futures

from baseline_constants import BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY
from utils.client_resource_utils import calculate_hardware_score, calculate_network_score, calculate_data_quality_score

class Server:
    
    def __init__(self, client_model):
        self.client_model = client_model
        self.model = client_model.get_params()
        self.selected_clients = []
        self.updates = []
        self.smoothing_factor = 0.2
        self.previous_aggregated_mean = None
        self.previous_aggregated_variance = None

    def train_model(self, num_epochs, batch_size=10, minibatch=None, clients=None, simulate_delays=True):
        if clients is None:
            clients = self.selected_clients

        print("number of clients server is calling train on...", len(clients))

        sys_metrics = {c.id: {BYTES_WRITTEN_KEY: 0, BYTES_READ_KEY: 0, LOCAL_COMPUTATIONS_KEY: 0} for c in clients}
        
        # Lists to hold times for each client
        download_times = []
        training_times = []
        upload_times = []

        global_params = self.model

        def train_client(c):
            self.updates = []
            c.model.set_params(self.model)
            comp, num_samples, update, elbo, variance, d_time, t_time, u_time = c.train(num_epochs, batch_size, minibatch, global_params, simulate_delays)
            with threading.Lock():
                sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
                sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
                sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
                self.updates.append((num_samples, update, elbo, variance))
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

    def update_model(self, method='fedprox'):
        if method == 'fedavg':
            self.aggregate_updates_fedavg()
        elif method == 'fedprox':
            self.aggregate_updates_fedprox()
        elif method == 'bayesian':
            self.aggregate_updates_bayesian()
        elif method == 'hierarchical_bayesian':
            self.aggregate_updates_hierarchical_bayesian()

    def aggregate_updates_fedavg(self):
        print("inside fedavg...")
        total_weight = 0.
        base = [np.zeros_like(v.numpy(), dtype=np.float32) for v in self.updates[0][1]]
        for (client_samples, client_model) in self.updates:
            total_weight += client_samples
            for i, v in enumerate(client_model):
                base[i] += (client_samples * v.numpy().astype(np.float32))
        averaged_soln = [v / total_weight for v in base]

        self.model = averaged_soln
        self.updates = []

    def aggregate_updates_fedprox(self):
        """Aggregate client updates using FedProx."""
        print("inside FedProx aggregation...")
        total_weight = 0.
        base = [np.zeros_like(v.numpy(), dtype=np.float32) for v in self.updates[0][1]]
        for (client_samples, client_model, _, _) in self.updates:
            total_weight += client_samples
            for i, v in enumerate(client_model):
                base[i] += (client_samples * v.numpy().astype(np.float32))
        averaged_soln = [v / total_weight for v in base]

        self.model = averaged_soln
        self.updates = []

    def aggregate_updates_bayesian(self):
        print("inside bayesian aggregation...")
        total_weight = 0.
        weighted_mean = [np.zeros_like(v.numpy(), dtype=np.float32) for v in self.updates[0][1]]
        weighted_variance = [np.zeros_like(v.numpy(), dtype=np.float32) for v in self.updates[0][1]]

        for (num_samples, client_model, elbo, variance) in self.updates:
            weight = elbo / np.mean(variance)
            total_weight += weight
            for i, (mean_update, var_update) in enumerate(zip(client_model, variance)):
                weighted_mean[i] += weight * mean_update.numpy().astype(np.float32)
                weighted_variance[i] += weight * var_update

        aggregated_mean = [mean / total_weight for mean in weighted_mean]
        aggregated_variance = [var / total_weight for var in weighted_variance]

        for i, variable in enumerate(self.model):
            variable.assign(aggregated_mean[i])

        self.updates = []

    # def aggregate_updates_hierarchical_bayesian(self):

    #     print("inside (simple) hierarchical bayesian aggregation ...")
        
    #     mean_updates = [update[1] for update in self.updates]
    #     variance_updates = [update[3] for update in self.updates]

    #     aggregated_mean = []
    #     aggregated_variance = []

    #     for i in range(len(mean_updates[0])):
    #         inv_variances = [1.0 / var[i] for var in variance_updates]
    #         total_inv_variance = np.sum(inv_variances)
    #         normalized_weights = [inv_var / total_inv_variance for inv_var in inv_variances]

    #         mean_sum = np.sum([normalized_weights[j] * mean_updates[j][i].numpy() for j in range(len(mean_updates))], axis=0)
    #         variance_sum = np.sum([normalized_weights[j] * variance_updates[j][i] for j in range(len(variance_updates))], axis=0)
            
    #         aggregated_mean.append(mean_sum)
    #         aggregated_variance.append(variance_sum)
        
    #     for i, variable in enumerate(self.model):
    #         variable.assign(aggregated_mean[i])

    #     self.updates = []

    def aggregate_updates_hierarchical_bayesian(self):
        print("inside hierarchical bayesian aggregation (with smoothing factor)...")
        
        mean_updates = [update[1] for update in self.updates]
        variance_updates = [update[3] for update in self.updates]

        aggregated_mean = []
        aggregated_variance = []

        for i in range(len(mean_updates[0])):
            inv_variances = [1.0 / (var[i] + 1e-8) for var in variance_updates]
            total_inv_variance = np.sum(inv_variances)
            normalized_weights = [inv_var / total_inv_variance for inv_var in inv_variances]

            mean_sum = np.sum([normalized_weights[j] * mean_updates[j][i].numpy() for j in range(len(mean_updates))], axis=0)
            variance_sum = np.sum([normalized_weights[j] * variance_updates[j][i] for j in range(len(variance_updates))], axis=0)

            if self.previous_aggregated_mean is not None:
                mean_sum = self.smoothing_factor * self.previous_aggregated_mean[i] + (1 - self.smoothing_factor) * mean_sum
                variance_sum = self.smoothing_factor * self.previous_aggregated_variance[i] + (1 - self.smoothing_factor) * variance_sum
            
            aggregated_mean.append(mean_sum)
            aggregated_variance.append(variance_sum)

        self.previous_aggregated_mean = aggregated_mean
        self.previous_aggregated_variance = aggregated_variance

        for i, variable in enumerate(self.model):
            variable.assign(aggregated_mean[i])

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

