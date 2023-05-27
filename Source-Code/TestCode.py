import pygame, sys, time, random, numpy as np

n = 1
p = 0.5
size = 1
x = np.random.binomial(n,p,size)

x = np.random.rayleigh(1)
x = np.random.exponential(1)
x = []
y = []
for i in range(1,144+1):
    x.append(i)

start_index = 0
end_index = 12

for i in range(1,13):
    y.append(x[start_index:end_index])
    start_index+=12
    end_index+=12
print(y)