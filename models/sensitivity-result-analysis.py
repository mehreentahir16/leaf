import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the updated sensitivity analysis results
updated_CIFAR_data = pd.read_pickle('sensitivity_analysis_results.pkl')

def generate_percentage_boxplots(data, weight_index, weight_name, ax_acc, ax_time, max_time):
    """
    Generate boxplots with accuracy as percentages and total training time, 
    adapted to the updated data structure.
    """
    # Extract weights, mean accuracies, and total training time
    weights = [item['weights'] for item in data]
    final_accuracies = [item['accuracies'][-1] for item in data]
    total_times = [item['total_training_time'] for item in data]
    varying_weight = [w[weight_index] for w in weights]
    
    # Create a DataFrame for easy plotting
    df = pd.DataFrame({
        'Weight': varying_weight,
        'Accuracy (%)': final_accuracies,  # Mean accuracy as a percentage
        'Total Training Time': total_times
    })
    
    # Boxplot for Accuracy
    sns.boxplot(x='Weight', y='Accuracy (%)', data=df, ax=ax_acc, palette="Blues", width=0.6)
    ax_acc.set_ylim(0, 100)
    ax_acc.set_xlabel('')
    if weight_index == 0:
        ax_acc.set_ylabel('Final Accuracy (%)', fontsize=14)
    else:
        ax_acc.set_ylabel('')
    ax_acc.tick_params(axis='both', which='major', labelsize=12)
    
    # Boxplot for Total Training Time
    sns.boxplot(x='Weight', y='Total Training Time', data=df, ax=ax_time, palette="Greens", width=0.6)
    ax_time.set_ylim(0, max_time)
    if weight_index == 0:
        ax_time.set_ylabel('Total Training Time (seconds)', fontsize=14)
    else:
        ax_time.set_ylabel('')
    ax_time.set_xlabel(f'{weight_name} Value', fontsize=14)
    ax_time.tick_params(axis='both', which='major', labelsize=12)

# Indices and names for weights remain unchanged
weights_indices = [0, 1, 2]
weight_names = ["Hardware Weight", "Network Weight", "Data Quality Weight"]

# Get the maximum total training time for setting a consistent y-axis scale across plots
max_total_training_time = max(item['total_training_time'] for item in updated_CIFAR_data)

# Plotting setup also remains unchanged
fig, axes = plt.subplots(2, 3, figsize=(12, 6))  # 2 rows (Accuracy, Training Time) and 3 columns (Weights)

# Generate boxplots for each weight, using total training time and mean accuracies
for j, (weight_index, weight_name) in enumerate(zip(weights_indices, weight_names)):
    generate_percentage_boxplots(updated_CIFAR_data, weight_index, weight_name, axes[0, j], axes[1, j], max_total_training_time)
    axes[0, j].tick_params(axis='x', labelrotation=45)  # Rotate x-axis labels for readability
    axes[1, j].tick_params(axis='x', labelrotation=45)

plt.tight_layout(pad=0.5)  # Adjust spacing between subplots for better readability
plt.savefig("results/plots/FEMNIST-sensitivity-analysis.png")
plt.show()
