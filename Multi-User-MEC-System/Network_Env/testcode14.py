import numpy as np  # For computations (if needed)

# Dictionary to store all arrays dynamically
policy_data = {}

# List of metric names (without the policy number)
metrics = [
    "fairness_index",
    "outage_probability",
    "failed_urllc_transmissions",
    "urllc_throughput",
    "urllc_arriving_packets",
    "urllc_dropped_packets_resource_allocation",
    "urllc_dropped_packets_channel_rate",
    "urllc_dropped_packets_channel_rate_normalized",
    "urllc_successful_transmissions",
    "urllc_code_blocks_allocation",
    "offloading_ratios"
]

# List of policies
policies = [3, 5, 6]

# Function to initialize all arrays for a given policy number
def initialize_policy_data(policy_number):
    for metric in metrics:
        key = f"{metric}_policy_{policy_number}_multiplexing"
        policy_data[key] = []  # Initialize as an empty list

# Initialize data storage for all policies
for policy in policies:
    initialize_policy_data(policy)

# Simulating task arrival rates from 0.1 to 0.5 (increments of 0.1)
task_arrival_rates = np.arange(0.1, 0.6, 0.1)  # Generates [0.1, 0.2, 0.3, 0.4, 0.5]

# Function to compute metrics (Replace with actual computations)
def compute_metrics(task_arrival_rate, policy_number):
    """
    Replace these dummy calculations with actual logic based on your scenario.
    """
    return {
        "fairness_index": np.random.uniform(0.5, 1.0),
        "outage_probability": np.random.uniform(0, 0.5),
        "failed_urllc_transmissions": np.random.randint(0, 100),
        "urllc_throughput": task_arrival_rate * np.random.uniform(1e6, 1e7),  # Simulating large values
        "urllc_arriving_packets": np.random.randint(50, 150),
        "urllc_dropped_packets_resource_allocation": np.random.randint(0, 20),
        "urllc_dropped_packets_channel_rate": np.random.randint(0, 10),
        "urllc_dropped_packets_channel_rate_normalized": np.random.uniform(0, 1),
        "urllc_successful_transmissions": np.random.randint(10, 100),
        "urllc_code_blocks_allocation": np.random.randint(1, 10),
        "offloading_ratios": np.random.uniform(0, 1)
    }

# Loop over task arrival rates and policies
for task_arrival_rate in task_arrival_rates:
    for policy in policies:
        metrics_values = compute_metrics(task_arrival_rate, policy)  # Compute values
        
        # Append computed values to respective arrays
        for metric, value in metrics_values.items():
            key = f"{metric}_policy_{policy}_multiplexing"
            policy_data[key].append(value)

# Print formatted output
for key, values in policy_data.items():
    formatted_values = ",".join(f"{v:.6f}" for v in values)  # Format each value to 6 decimal places
    print(f"{key} = [{formatted_values}]")
