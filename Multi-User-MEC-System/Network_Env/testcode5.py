import math as m

def square_sum(array):
    sum = 0
    for i in array:
        sum+=m.pow(i,2)

    return sum

number_of_users = 5
throughputs = [3, 1, 2, 2, 2]
sum_ = square_sum(throughputs)
print(sum)
index = (m.pow(sum(throughputs),2))/(len(throughputs)*sum_)
print(index)

f = 3

d = [x + f for x in throughputs]
print(d)

