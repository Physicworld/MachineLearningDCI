import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
plt.style.use('classic')

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

iris_dataset = datasets.load_iris()
X = iris_dataset.data
y = iris_dataset.target
label_strings = iris_dataset.target_names

# APLICANDO PCA PARA REDUCIR A 3D.
pca_model = PCA(n_components=3)

X_3D = pca_model.fit_transform(X)

# APLICANDO PCA PARA REDUCIR A 2D.

pca_model = PCA(n_components=2)

X_2D = pca_model.fit_transform(X)


# APLICANDO MODELO KMEANS PARA LOS 3 DATASET

kmeans_model = KMeans(n_clusters=3)

labels_original = kmeans_model.fit_predict(X)
print("ORIGINAL CLUSTER CENTERS")
print(kmeans_model.cluster_centers_)

labels_3D = kmeans_model.fit_predict(X_3D)
print("3D CLUSTER CENTERS")
print(kmeans_model.cluster_centers_)
fig = plt.figure()
ax1 = fig.add_subplot(111, projection = '3d')
ax1.scatter(X_3D[:,0], X_3D[:,1], X_3D[:,2], c = labels_3D)
plt.show()

labels_2D = kmeans_model.fit_predict(X_2D)
print("2D CLUSTER CENTERS")
print(kmeans_model.cluster_centers_)
plt.scatter(x=X_2D[:,0], y = X_2D[:,1], c = labels_2D)
plt.show()