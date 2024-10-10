# Reimport necessary libraries after reset
import scipy.stats as stats

def compute_probability(N, p, l_u, achieved_data_rates):
    """
    This function computes the probability that the sum of achieved data rates
    is less than or equal to the product of the URLLC packet size and the number of arriving packets.
    
    Parameters:
    - N: number of URLLC users (number of trials in binomial distribution)
    - p: probability of packet arrival for each user (success probability in binomial distribution)
    - l_u: URLLC user's packet size (a fixed constant)
    - achieved_data_rates: list of achieved data rates for each URLLC user
    
    Returns:
    - probability: the computed probability
    """
    # Compute the total achieved data rate
    total_achieved_data_rate = sum(achieved_data_rates)
    
    # Compute the CDF of the binomial distribution for the number of arriving packets
    L = stats.binom(N, p)
    
    # Compute the CDF value for the achieved data rate divided by the packet size
    probability = 1 - L.cdf(total_achieved_data_rate / l_u)
    
    return probability

# Example parameters
N = 10  # Number of URLLC users
p = 0.5  # Packet arrival probability
l_u = 5  # Packet size
achieved_data_rates = [4, 0, 0, 4, 0, 6, 4, 0, 6, 4]  # Example achieved data rates for each user

# Compute the probability
probability = compute_probability(N, p, l_u, achieved_data_rates)
print(probability)
