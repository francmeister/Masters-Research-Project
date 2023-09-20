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
throughputs = [4, 7, 1, 3]

#for array in subcarrier_actions:
sum_ = square_sum(throughputs)
print(sum)
index = (m.pow(sum(throughputs),2))/(len(throughputs)*sum_)
print('index: ', index)


