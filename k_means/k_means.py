import matplotlib.pyplot as plt
import mglearn as mglearn
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def execute():
    X, y = make_blobs(random_state=1)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    print("Cluster membership:\n{}".format(kmeans.labels_))

    fig = plt.figure()
    mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
    mglearn.discrete_scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2], markers='^',
                             markeredgewidth=2)
    fig.savefig('k_means/3_means.png')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(X)
    assignments = kmeans.labels_
    mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[0])

    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)
    assignments = kmeans.labels_
    mglearn.discrete_scatter(X[:, 0], X[:, 1], assignments, ax=axes[1])

    fig.savefig('k_means/2,5-means.png')
