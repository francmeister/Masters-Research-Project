import math as m
import numpy as np
subcarrier_actions = np.load('subcarrier_actions.npy')
print(len(subcarrier_actions))
def square_sum(array):
    sum = 0
    for i in array:
        sum+=m.pow(i,2)

    return sum

number_of_users = 5
throughputs = [3, 1, 2, 2, 2]
for array in subcarrier_actions:
    sum_ = square_sum(array)
    print(sum)
    index = (m.pow(sum(array),2))/(len(array)*sum_)
    print('index: ', index)


