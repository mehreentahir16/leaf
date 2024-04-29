import pickle

def extract_details(file_path):
    # Load the data from the file
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    
    # Extract 'time_to_reach_accuracy_thresholds' and convert to seconds
    time_to_reach_accuracy_thresholds = data.get('time_to_reach_accuracy_thresholds', {})
    time_to_reach_accuracy_thresholds_seconds = {
        accuracy: time if time is not None else None
        for accuracy, time in time_to_reach_accuracy_thresholds.items()
    }
    
    # Calculate the number of rounds for each accuracy threshold reached
    accuracies = data.get('accuracies', [])
    rounds_to_reach_accuracy_thresholds = {}
    for accuracy, time in time_to_reach_accuracy_thresholds.items():
        for index, acc in enumerate(accuracies):
            if acc >= accuracy:
                rounds_to_reach_accuracy_thresholds[accuracy] = index + 1
                break
    
    final_accuracy = accuracies[-1] if accuracies else None
    # Extract total training time and convert to seconds
    total_training_time_seconds = data.get('training_time', 0)
    number_of_clients = data.get('number_of_clients', 0)  
    avg_number_of_samples = data.get('avg_number_of_samples', 0)
    total_unique_training_samples = data.get('total_unique_training_samples', 0)
    
    return rounds_to_reach_accuracy_thresholds, time_to_reach_accuracy_thresholds_seconds, total_training_time_seconds, final_accuracy, number_of_clients, avg_number_of_samples, total_unique_training_samples

# Use the path to your new file here
file_path_new = 'results/data_femnist_selection_random_clients_20_budget_50_results_test.pkl'
rounds_info, time_info, total_training_time_seconds, final_accuracy, number_of_clients, avg_number_of_samples, total_unique_training_samples = extract_details(file_path_new)

print("Rounds to reach accuracy levels:", rounds_info)
print("Time (in seconds) to reach accuracy levels:", time_info)
print("Total training time in seconds:", total_training_time_seconds)
print("Final accuracy:", final_accuracy)
print("number_of_clients", number_of_clients)
print("avg_number_of_samples", avg_number_of_samples)
print("total_unique_training_samples", total_unique_training_samples)
