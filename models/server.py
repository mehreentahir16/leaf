import threading
import numpy as np

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
            comp, num_samples, update, mean, variance, d_time, t_time, u_time = c.train(num_epochs, batch_size, minibatch, simulate_delays)
            with threading.Lock():
                sys_metrics[c.id][BYTES_READ_KEY] += c.model.size
                sys_metrics[c.id][BYTES_WRITTEN_KEY] += c.model.size
                sys_metrics[c.id][LOCAL_COMPUTATIONS_KEY] = comp
                self.updates.append((num_samples, update, mean, variance))
                # Store times for each client
                download_times.append(d_time)
                training_times.append(t_time)
                upload_times.append(u_time)

        threads = [threading.Thread(target=train_client, args=(c,)) for c in clients]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Use the maximum time spent in any operation across all clients as the simulated time for that operation
        total_download_time = max(download_times) if download_times else 0
        total_training_time = max(training_times) if training_times else 0
        total_upload_time = max(upload_times) if upload_times else 0

        return sys_metrics, total_download_time, total_training_time, total_upload_time

    def update_model(self, method='poe'):
        if method == 'fedavg':
            self.aggregate_updates_fedavg()
        elif method == 'weighted_sum':
            self.aggregate_updates_weighted_sum()
        elif method == 'poe':
            self.aggregate_updates_poe()

    def aggregate_updates_fedavg(self):
        total_weight = 0.
        base = [0] * len(self.updates[0][1])
        for (client_samples, client_model, _, _) in self.updates:
            total_weight += client_samples
            for i, v in enumerate(client_model):
                base[i] += (client_samples * v.astype(np.float64))
        averaged_soln = [v / total_weight for v in base]

        self.model = averaged_soln
        self.updates = []

    def aggregate_updates_weighted_sum(self):
        if not self.updates:
            raise ValueError("No updates available for aggregation")
        
        total_weight = 0.
        base_mean = None
        base_variance = None


        for update in self.updates:
            client_samples, _, mean, variance = update
            if mean is None or variance is None:
                raise ValueError("Encountered update from a client with None values")

            if base_mean is None:
                base_mean = [0] * len(mean)
                base_variance = [0] * len(variance)

            total_weight += client_samples
            for i, (m, v) in enumerate(zip(mean, variance)):
                base_mean[i] += client_samples * m
                base_variance[i] += client_samples * (v + m**2)

        if total_weight == 0:
            raise ValueError("No valid updates to aggregate")

        averaged_mean = [bm / total_weight for bm in base_mean]
        averaged_variance = [(bv / total_weight) - (am**2) for bv, am in zip(base_variance, averaged_mean)]

        self.model = averaged_mean
        self.updates = []

    def aggregate_updates_poe(self):
        if not self.updates:
            raise ValueError("No updates available for aggregation")
        
        print("length of updates in poe", len(self.updates))
        print("updates shape", np.shape(self.updates))
        log_posterior_sum = None
        for update in self.updates:
            client_samples, _, mean, variance = update
            print("mean length in poe", len(mean))
            print("variance length in poe", len(variance))
            if mean is None or variance is None:
                raise ValueError("Encountered update from a client with None values")

            # Ensure variance is a numpy array and handle element-wise operations properly
            mean = np.array(mean)
            variance = np.array(variance).flatten()
            print(f"Client samples: {client_samples}, Mean shape: {np.shape(mean)}, Variance shape: {np.shape(variance)}")

            # Ensure variance is positive
            # if (variance <= 0).any():
            #     variance = np.where(variance <= 0, 1e-10, variance)
            # print(f"Adjusted Variance shape: {np.shape(variance)}, Values: {variance}")

            # Generate samples
            try:
                sqrt_variance = np.sqrt(variance)
                print(f"sqrt_variance shape: {sqrt_variance.shape}, Values: {sqrt_variance}")
                samples = np.random.normal(loc=mean, scale=sqrt_variance.reshape(-1, 1), size=(len(mean), client_samples))
            except Exception as e:
                print(f"Exception during sample generation: {e}")
                continue
            print(f"Posterior samples shape: {samples.shape}")

            # Compute log posterior
            log_posterior = np.sum(np.log(samples + 1e-10), axis=0)  # Adding epsilon to prevent log(0)
            if log_posterior_sum is None:
                log_posterior_sum = log_posterior
            else:
                log_posterior_sum += log_posterior

        if log_posterior_sum is None:
            raise ValueError("No valid updates to aggregate")

        normalized_posterior = np.exp(log_posterior_sum - np.max(log_posterior_sum))
        normalized_posterior /= np.sum(normalized_posterior)
        self.model = normalized_posterior.tolist()
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
        model_sess =  self.client_model.sess
        return self.client_model.saver.save(model_sess, path)

    def close_model(self):
        self.client_model.close()
