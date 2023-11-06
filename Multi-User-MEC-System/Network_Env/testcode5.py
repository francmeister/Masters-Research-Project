import numpy as np

channel_gain_max = 50
number_of_users = 3
number_of_RB = 7
battery_energy_max = 20
number_of_batteries_per_user = 1
number_of_states_per_user = number_of_RB + number_of_batteries_per_user

x = np.array([[[channel_gain_max for _ in range(number_of_RB)] + [battery_energy_max for _ in range(1)]]*number_of_users ],dtype=np.float32)

x = x.reshape(number_of_users ,number_of_states_per_user)
print(x)