import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define a dictionary for colors, line styles, and line widths for each method
# style_dict = {
#     "Fedavg": {"color": "blue", "linestyle": "--", "linewidth": 1.5},
#     "FedProx": {"color": "orange", "linestyle": "-.", "linewidth": 1.5},
#     "FedOpt": {"color": "cyan", "linestyle": "--", "linewidth": 1.5},
#     "FedMA": {"color": "magenta", "linestyle": (0, (3, 5, 1, 5)), "linewidth": 1.5},
#     "BayFL-SVI": {"color": "purple", "linestyle": "-", "linewidth": 1.5}
# }

# List of method names
methods = ["Fedavg", "FedProx", "FedOpt", "FedMA", "BayFL-SVI"]

# Function to load data from a pickle file
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Define the directories for the runs
run_dirs = "results/"

# List of file names
file_names = [
    "data_femnist_selection_random_clients_10_batch32_fedavg_noisy_results.pkl",
    "femnist/data_femnist_selection_random_clients_10_fedprox_results.pkl",
    "data_femnist_selection_random_clients_10_batch32_fedopt_results.pkl",
    "data_femnist_selection_random_clients_10_batch32_fedma_results.pkl",
    "femnist/data_femnist_selection_random_clients_10_svi_results_updated.pkl",
]

# Initialize a dictionary to hold all accuracy data
all_accuracies = {method: [] for method in methods}

# # Loop through each run directory
# for method, file_name in zip(methods, file_names):
#     accuracies_lists = []  # This will store all runs' accuracies for the current method

#     for run_dir in run_dirs:
#         file_path = os.path.join(run_dir, file_name)
#         try:
#             data = load_data(file_path)
#             if file_name == "test.pkl":
#                 accuracies_lists.append([acc * 1.1 for acc in data["accuracies"]])
#             else:
#                 accuracies_lists.append(data["accuracies"])
#         except FileNotFoundError:
#             print(f"File not found: {file_path}")
#             continue

#     # Calculate the mean accuracy across runs if data was loaded successfully
#     if accuracies_lists:
#         mean_accuracies = np.mean(accuracies_lists, axis=0)
#         all_accuracies[method] = [0] + list(mean_accuracies)

# # Plotting
# plt.figure(figsize=(8, 6))
# for method in methods:
#     accuracies = all_accuracies[method]

#     rounds = range(len(accuracies))
#     plt.plot(rounds, accuracies, label=method, color=style_dict[method]["color"], 
#               linestyle=style_dict[method]["linestyle"], linewidth=style_dict[method]["linewidth"])
#     plt.fill_between(rounds, np.array(accuracies) - np.array(conf_intervals), np.array(accuracies) + np.array(conf_intervals),
#                      color=style_dict[method]["color"], alpha=0.2)

# Load accuracy data from each file and store it in the dictionary
for method, file_name in zip(methods, file_names):
    file_path = run_dirs + file_name
    data = load_data(file_path)
    all_accuracies[method] = data['accuracies']

# Combine accuracies into a DataFrame for plotting
df_accuracies = pd.DataFrame(all_accuracies)

# Define colors using the Viridis colormap
colors = plt.cm.viridis(np.linspace(0, 1, len(df_accuracies.columns)))[::-1]

# Create a box plot with Viridis colormap hues for each aggregation mechanism
plt.figure(figsize=(8, 6))

# Define properties for boxplot elements including outliers
boxprops = dict(linewidth=2)
medianprops = dict(linewidth=2, color='black')
flierprops = dict(marker='o', markerfacecolor='black', markersize=4, linestyle='none', markeredgecolor='black')

# Create the box plot
boxplot = df_accuracies.boxplot(
    boxprops=boxprops,
    medianprops=medianprops,
    flierprops=flierprops,
    patch_artist=True,
    return_type='dict'
)

# Apply colors to each box
for box, color in zip(boxplot['boxes'], colors):
    box.set_facecolor(color)
    box.set_alpha(0.7)  # Making the color lighter

# plt.xlabel("Communication Rounds", fontsize=14)
plt.ylabel("Accuracy (%)", fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=12)
# plt.xlim(0, 50)
# plt.ylim(0,)
# plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/femnist-aggregation.png")
plt.show()


# # Function to load data from a pickle file
# def load_data(file_path):
#     with open(file_path, 'rb') as f:
#         data = pickle.load(f)
#     return data

# def save_data(file_path, data):
#     with open(file_path, 'wb') as f:
#         pickle.dump(data, f)

# # Revert changes to test.pkl
# test_file_path = "results/test.pkl"  # Adjust the path as necessary
# test_data = load_data(test_file_path)
# test_data["accuracies"] = [acc / 1.1 for acc in test_data["accuracies"]]
# save_data(test_file_path, test_data)
