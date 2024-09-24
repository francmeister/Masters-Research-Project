import scipy.stats as stats
import statistics
import math

a = 13.8
mean = 14
std = 1
f = stats.norm.cdf(a,mean,std)
rates = [100, 200, 150, 130]
#variance = statistics.pvariance(rates)


def variance_calculation(rates):
    average = sum(rates)/len(rates)
    var = 0
    for sample in rates:
        var = var + math.pow((sample-average),2)

    variance = var/len(rates)

    return variance


variance = variance_calculation(rates)
print(math.factorial(3))