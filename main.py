import random
import scipy.stats as stats
import numpy



bandit_number = 10
trials = 100
bandits = []
means = []
expectations = []
gittins = []
disc_sum = 0
min, max = 0, 1
sigma = 0.1
disc_fact = 0.3
switch_cost = 0.05

#Initializing
for i in range(0, bandit_number):

    rand_mean = random.uniform(0.0, 1.0)
    means.append(rand_mean)
    dist = stats.truncnorm((min - means[i]) / sigma, (max - means[i]) / sigma, loc=means[i], scale=sigma)
    bandit = dist.rvs(trials)
    bandits.append(bandit)
    expectations.append(0)


#    print("bandit: "+ str(bandit))
#    print("mean: " + str(rand_mean))
#Running trials
for i in range(0, trials):
    #Calculating gittins index
    gittins = []
    for j in range(0, bandit_number):
        expectations[j] += bandits[j][i] * disc_fact**trials
        disc_sum += disc_fact**trials
        gittins.append(expectations[j]/disc_sum)
    gittins_idx = numpy.array(gittins)
    max_ = numpy.amax(gittins_idx)
    result = numpy.where(gittins_idx == max_)
    idx_ = result[0][0]
    print("The bandit number "+str(idx_)+" was chosen and its mean is: "+str(means[idx_])+" and its expectation is:" +str(max_))
