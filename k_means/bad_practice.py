import matplotlib.pyplot as plt
import mglearn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def execute():
    X_varied, y_varied = make_blobs(n_samples=200,
                                    cluster_std=[1.0, 2.5, 0.5],
                                    random_state=170)
    y_pred = KMeans(n_clusters=3, random_state=0).fit_predict(X_varied)

    fig = plt.figure()
    mglearn.discrete_scatter(X_varied[:, 0], X_varied[:, 1], y_pred)
    plt.legend(["cluster 0", "cluster 1", "cluster2"], loc='best')
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    fig.savefig('k_means/bad_practice_concentration.png')

    X, y = make_blobs(random_state=170, n_samples=600)
    rng = np.random.RandomState(74)

    transformation = rng.normal(size=(2, 2))
    X = np.dot(X, transformation)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)

    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=mglearn.cm3)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='^', c=[0, 1, 2], linewidths=2,
                cmap=mglearn.cm3)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    fig.savefig('k_means/bad_practice_not_rounded.png')


