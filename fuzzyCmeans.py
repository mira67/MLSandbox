"""
Fuzzy C Means Example using skfuzzy lib
Author: mira67
Date: 08/30/2016
"""
import matplotlib.pyplot as plt
import numpy as np
import skfuzzy as fuzz

# Step 1: initialize data and labels
mu1 = [2, 0]
sigma1 = [[1, 0], [0, 1]]

mu2 = [-2, 0]
sigma2 = [[1, 0], [0, 1]]

dlabel1 = np.ones(50)
dlabel2 = np.ones(50) * 2

x1, y1 = np.random.multivariate_normal(mu1, sigma1, 50).T
x2, y2 = np.random.multivariate_normal(mu2, sigma2, 50).T

x = np.concatenate((x1, x2), axis=0)
y = np.concatenate((y1, y2), axis=0)

dlabel = np.concatenate((dlabel1, dlabel2), axis=0)

# Step 2: plot data
colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

plt.figure(1)
plt.subplot(121)

plt.scatter(x, y, c=dlabel,s=100)
#plt.plot(x, y, '+',mew=2, ms=8)
# plt.axis('equal')

# Step 3: Fuzzy C Means Example
ncenters = 2
alldata = np.vstack((x, y))
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    alldata, ncenters, 2, error=0.005, maxiter=100, init=None)

# Step 4: Visualize Clustering Results
fpcs = []
fpcs.append(fpc)

cluster_membership = np.argmax(u, axis=0)

plt.subplot(122)

plt.scatter(x, y, c=cluster_membership,s=100)

  # Mark the center of each fuzzy cluster
for pt in cntr:
	plt.plot(pt[0], pt[1], 'rs',mew=2, ms=8)

plt.show()
