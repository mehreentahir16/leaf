import random
import argparse

DATASETS = ['sent140', 'femnist', 'shakespeare', 'celeba', 'synthetic', 'reddit']
SIM_TIMES = ['small', 'medium', 'large']
CLIENT_SELECTION_STRATEGIES = ['random', 'greedy', 'price_based', 'resource_based', 'active', 'pow-d', 'promethee']  
ATTACKS=['label-flip', 'random-noise']

def generate_random_flip_pairs(num_labels):
    labels = list(range(num_labels))
    flipped_labels = labels.copy()
    random.shuffle(flipped_labels)
    
    # Ensure no label maps to itself
    for i in range(num_labels):
        if flipped_labels[i] == labels[i]:
            # Swap with the next label or wrap around
            swap_idx = (i + 1) % num_labels
            flipped_labels[i], flipped_labels[swap_idx] = flipped_labels[swap_idx], flipped_labels[i]
    
    return {labels[i]: flipped_labels[i] for i in range(num_labels)}

label_flipping_config = {
    'femnist': {
        'flip_fraction':0.5,
        'target_labels': list(range(62)),  # Representing all classes
        'flip_pairs': generate_random_flip_pairs(62)
    },
    'celeba': {
        'flip_fraction': 0.5,
        'target_labels': [0, 1],  # 0: Not Smiling, 1: Smiling
        'flip_pairs': {0: 1, 1: 0}
    },
    'synthetic': {
        'flip_fraction': 0.3,
        'target_labels': list(range(5)),  # Assuming a 10-class synthetic dataset
        'flip_pairs': generate_random_flip_pairs(5)
    },
    # Add other datasets as needed
}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    required=True)
    parser.add_argument('-model',
                    help='name of model;',
                    type=str,
                    required=True)
    parser.add_argument('--num-rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=-1)
    parser.add_argument('--eval-every',
                    help='evaluate every ____ rounds;',
                    type=int,
                    default=1)
    parser.add_argument('--clients-per-round',
                    help='number of clients trained per round;',
                    type=int,
                    default=-1)
    parser.add_argument('--client-selection-strategy',
                    help='strategy to select clients each round',
                    type=str,
                    choices=CLIENT_SELECTION_STRATEGIES,
                    default='random')
    parser.add_argument('--malicious-fraction',
                    help='Fraction of clients that are malicious (for testing purposes)',
                    type=float,
                    default=0.2)
    parser.add_argument('--attack',
                    help="Type of attack to simulate",
                    type=str, 
                    choices=ATTACKS,
                    default='label_flip')
    parser.add_argument('--batch-size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=10)
    parser.add_argument('--seed',
                    help='seed for random client sampling and batch splitting',
                    type=int,
                    default=0)
    parser.add_argument('--metrics-name', 
                    help='name for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)
    parser.add_argument('--metrics-dir', 
                    help='dir for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)
    parser.add_argument('--use-val-set', 
                    help='use validation set;', 
                    action='store_true')

    # Minibatch doesn't support num_epochs, so make them mutually exclusive
    epoch_capability_group = parser.add_mutually_exclusive_group()
    epoch_capability_group.add_argument('--minibatch',
                    help='None for FedAvg, else fraction;',
                    type=float,
                    default=None)
    epoch_capability_group.add_argument('--num-epochs',
                    help='number of epochs when clients train on data;',
                    type=int,
                    default=2)

    parser.add_argument('-t',
                    help='simulation time: small, medium, or large;',
                    type=str,
                    choices=SIM_TIMES,
                    default='large')
    parser.add_argument('-lr',
                    help='learning rate for local optimizers;',
                    type=float,
                    default=-1,
                    required=False)

    return parser.parse_args()
