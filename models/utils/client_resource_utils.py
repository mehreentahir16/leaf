import numpy as np

def estimate_network_delay(model_size_bytes, bandwidth_mbps, latency_ms):
    """
    Estimates network based on model size, client's download bandwidth, and latency.

    Args:
        model_size_bytes: Size of the model in bytes.
        bandwidth_mbps: Client's download bandwidth in Mbps.
        latency_ms: Network latency in milliseconds.

    Returns:
        time in seconds.
    """
    bandwidth_bytes_per_sec = bandwidth_mbps * 125000  # Convert Mbps to bytes/sec
    delay = model_size_bytes / bandwidth_bytes_per_sec + latency_ms / 1000.0
    return delay

def estimate_training_time(comp, cpu_cores, cpu_frequency_ghz, cpu_utilization, total_memory, available_memory):
    """
    Estimates the training time based on computational workload, CPU characteristics, and memory impact.

    Args:
        comp: Total floating point operations (FLOPs) for the training.
        cpu_cores: Number of CPU cores.
        cpu_frequency_ghz: CPU frequency in GHz.
        cpu_utilization: Expected CPU utilization percentage during training.
        total_memory: Total system memory in GB.
        available_memory: Available system memory in GB for training.

    Returns:
        Estimated training time in seconds.
    """
    adjusted_processing_capacity = cpu_cores * cpu_frequency_ghz * (cpu_utilization/100)
    memory_availability_factor = available_memory / total_memory
    memory_impact_factor = 1 / memory_availability_factor

    training_time_seconds = (comp / (adjusted_processing_capacity * 1e9 )) * memory_impact_factor

    return training_time_seconds

def computation_utility(cpu_count, cpu_cores, cpu_frequency, gpu_presence, gamma=1):
    """Calculate the utility of CPU (including count) and GPU."""
    return (cpu_count * cpu_cores * cpu_frequency) * (1 + gamma * gpu_presence)


def memory_utility(ram):
    """Calculate the utility of RAM, favoring more RAM."""
    return np.sqrt(ram)  # Square root function to favor higher RAM values more aggressively

def storage_utility(storage):
    """Calculate the utility of storage."""
    return np.log1p(storage)  # Logarithmic function for diminishing returns on additional storage

def calculate_hardware_score(cpu_count, cpu_cores, cpu_frequency, gpu_presence, ram, storage, w_comp=0.5, w_ram=0.4, w_storage=0.1):
    comp_utility = computation_utility(cpu_count, cpu_cores, cpu_frequency, gpu_presence)
    ram_utility_val = memory_utility(ram)
    storage_utility_val = storage_utility(storage)
    return w_comp * comp_utility + w_ram * ram_utility_val + w_storage * storage_utility_val

def calculate_network_score(bandwidth, latency, w_bandwidth=0.7, w_responsiveness=0.3):
    responsiveness = 1/(latency)
    return w_bandwidth * bandwidth + w_responsiveness * responsiveness

def calculate_data_quality_score(data_size, loss, w_size=0.7, w_loss=0.3):
    return w_size * data_size + w_loss * loss
    # return data_size/cost
