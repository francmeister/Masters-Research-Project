import numpy as np
import matplotlib.pyplot as plt
from numpy import interp

# A total of 12 urllc users
one_embb_user = [0.000006,0.003152,0.085441, 0.550993,0.990099]
three_embb_user = [0.000004,0.002257, 0.068737,0.492101, 0.990099]
five_embb_user = []
seven_embb_user = [0.000005,0.002661,0.077574, 0.531721,0.990099]
nine_embb_user = [0.000018,0.005686, 0.115723, 0.607640, 0.990099]
prob_gen_task = [0.2,0.4,0.6,0.8,1]