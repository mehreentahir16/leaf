import time
import random
import warnings
from utils.model_utils import get_model_size, get_update_size
from utils.client_resource_utils import estimate_network_delay, estimate_training_time

class Client:
    
    def __init__(self, client_id, group=None, train_data={'x' : [],'y' : []}, eval_data={'x' : [],'y' : []}, model=None):
        self._model = model
        self.id = client_id
        self.group = group
        self.train_data = train_data
        self.eval_data = eval_data
        self.hardware_config = self.assign_hardware()
        self.network_config = self.assign_network()

    def assign_hardware(self):
        """Randomly assigns detailed hardware configuration, including separate RAM values."""
        categories = {
            'Low-End': {'CPU Count': 1, 'Cores': 1, 'Frequency': 1.2, 'CPU Utilization': random.uniform(43, 100), 'GPU': 0, 'RAM': random.randint(1, 2), 'Available RAM': random.uniform(0.5, 1), 'Storage': random.uniform(1, 4)},  # GHz, GB for RAM and Storage
            'Mid-Range': {'CPU Count': 1, 'Cores': 2, 'Frequency': 2.5, 'CPU Utilization': random.uniform(25, 49), 'GPU': 0, 'RAM': random.randint(2, 4), 'Available RAM': random.uniform(1, 3), 'Storage': random.uniform(4, 8)},  # GHz, GB for RAM and Storage
            'High-End': {'CPU Count': 2, 'Cores': 4, 'Frequency': 3.5, 'CPU Utilization': random.uniform(12, 35), 'GPU': 0, 'RAM': random.randint(4, 8), 'Available RAM': random.uniform(3, 7), 'Storage': random.uniform(8, 64)},  # GHz, GB for RAM and Storage
            'Excellent': {'CPU Count': 4, 'Cores': 8, 'Frequency': 3.5, 'CPU Utilization': random.uniform(6, 36), 'GPU': 1, 'RAM': random.randint(8, 64), 'Available RAM': random.uniform(5, 64), 'Storage': random.uniform(32, 64)}  # GHz, GB for RAM and Storage
        }
    
        # Adjusting probabilities to make 'Excellent' category rare
        choices, weights = zip(*[
            ('Low-End', 0.3),  # 30% chance
            ('Mid-Range', 0.45),  # 45% chance
            ('High-End', 0.2),  # 20% chance
            ('Excellent', 0.05)  # 5% chance, making it rare
        ])
        selected_category = random.choices(population=choices, weights=weights, k=1)[0]
        return categories[selected_category]

    def assign_network(self):
        """Randomly assigns network characteristics, allowing high-end devices to have poor network and vice versa."""
        conditions = {
            'Poor': {'Bandwidth': random.uniform(1, 4), 'Latency': random.randint(20, 100)},
            'Average': {'Bandwidth': random.uniform(4, 10), 'Latency': random.randint(20, 80)},
            'Good': {'Bandwidth': random.uniform(10, 100), 'Latency': random.uniform(5,50)},
            'Excellent': {'Bandwidth': random.uniform(100, 1000), 'Latency': random.randint(1, 10)}
        }
        return random.choice(list(conditions.values()))

    def train(self, num_epochs, batch_size, minibatch, simulate_delays=True):
        """Trains on self.model using the client's train_data.

        Args:
            num_epochs: Number of epochs to train. Unsupported if minibatch is provided (minibatch has only 1 epoch)
            batch_size: Size of training batches.
            minibatch: fraction of client's data to apply minibatch sgd,
                None to use FedAvg
        Return:
            comp: number of FLOPs executed in training process
            num_samples: number of samples used in training
            update: set of weights
            update_size: number of bytes in update
        """
        download_time = 0
        training_time = 0
        upload_time = 0
        if simulate_delays==True: 
            untrained_model_size = get_model_size(self.model)
            download_time = estimate_network_delay(untrained_model_size, self.network_config['Bandwidth'], self.network_config['Latency'])
            # time.sleep(download_time)

            if minibatch is None:
                data = self.train_data
                comp, update = self.model.train(data, num_epochs, batch_size)
            else:
                frac = min(1.0, minibatch)
                num_data = max(1, int(frac*len(self.train_data["x"])))
                xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
                data = {'x': xs, 'y': ys}

                # Minibatch trains for only 1 epoch - multiple local epochs don't make sense!
                num_epochs = 1
                comp, update = self.model.train(data, num_epochs, num_data)
            
            training_time = estimate_training_time(comp, self.hardware_config['CPU Count']*self.hardware_config['Cores'], self.hardware_config['Frequency'], self.hardware_config['CPU Utilization'], self.hardware_config['RAM'], self.hardware_config['Available RAM'])
            # time.sleep(training_time)
            update_size = get_update_size(update)
            upload_time = estimate_network_delay(update_size, self.network_config['Bandwidth'], self.network_config['Latency'])
            # time.sleep(upload_time)

        else: 

            if minibatch is None:
                data = self.train_data
                comp, update = self.model.train(data, num_epochs, batch_size)
            else:
                frac = min(1.0, minibatch)
                num_data = max(1, int(frac*len(self.train_data["x"])))
                xs, ys = zip(*random.sample(list(zip(self.train_data["x"], self.train_data["y"])), num_data))
                data = {'x': xs, 'y': ys}

                # Minibatch trains for only 1 epoch - multiple local epochs don't make sense!
                num_epochs = 1
                comp, update = self.model.train(data, num_epochs, num_data)
        
        num_train_samples = len(data['y'])
        return comp, num_train_samples, update, download_time, training_time, upload_time

    def test(self, set_to_use='test'):
        """Tests self.model on self.test_data.
        
        Args:
            set_to_use. Set to test on. Should be in ['train', 'test'].
        Return:
            dict of metrics returned by the model.
        """
        assert set_to_use in ['train', 'test', 'val']
        if set_to_use == 'train':
            data = self.train_data
        elif set_to_use == 'test' or set_to_use == 'val':
            data = self.eval_data
        return self.model.test(data)

    @property
    def num_test_samples(self):
        """Number of test samples for this client.

        Return:
            int: Number of test samples for this client
        """
        if self.eval_data is None:
            return 0
        return len(self.eval_data['y'])

    @property
    def num_train_samples(self):
        """Number of train samples for this client.

        Return:
            int: Number of train samples for this client
        """
        if self.train_data is None:
            return 0
        return len(self.train_data['y'])

    @property
    def num_samples(self):
        """Number samples for this client.

        Return:
            int: Number of samples for this client
        """
        train_size = 0
        if self.train_data is not None:
            train_size = len(self.train_data['y'])

        test_size = 0 
        if self.eval_data is not  None:
            test_size = len(self.eval_data['y'])
        return train_size + test_size

    @property
    def model(self):
        """Returns this client reference to model being trained"""
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model
