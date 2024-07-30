import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_aggregation_behavior(aggregation_data_file='aggregation_data.pkl'):
    with open(aggregation_data_file, 'rb') as f:
        data = pickle.load(f)
    
    mean_history = np.array(data['means_history'])
    variance_history = np.array(data['variances_history'])

    # Plot the aggregated means over rounds
    plt.figure(figsize=(12, 6))
    for i in range(mean_history.shape[1]):
        plt.plot(mean_history[:, i], label=f'Mean Parameter {i}')
    plt.title('Aggregated Means Over Rounds')
    plt.xlabel('Rounds')
    plt.ylabel('Mean Values')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot the aggregated variances over rounds
    plt.figure(figsize=(12, 6))
    for i in range(variance_history.shape[1]):
        plt.plot(variance_history[:, i], label=f'Variance Parameter {i}')
    plt.title('Aggregated Variances Over Rounds')
    plt.xlabel('Rounds')
    plt.ylabel('Variance Values')
    plt.legend()
    plt.grid(True)
    plt.show()
