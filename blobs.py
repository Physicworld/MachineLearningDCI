import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs

plt.figure()
X, y_true = make_blobs(n_samples=50, centers=3, cluster_std=.9, random_state=3)
plt.scatter(X[:,0], X[:,1], s=50)
plt.show()