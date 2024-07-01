import math


noise_spectral_density_dbm = -174 # -174dBM/Hz
noise_spectral_density_W = (math.pow(10,(noise_spectral_density_dbm/10)))/1000

print(noise_spectral_density_W)