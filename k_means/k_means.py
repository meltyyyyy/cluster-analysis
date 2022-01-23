import matplotlib.pyplot as plt
import mglearn as mglearn
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs, make_moons


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

    X, y = make_moons(n_samples=200, random_state=1, noise=0.05)

    kmeans = KMeans(n_clusters=10, random_state=0)
    kmeans.fit(X)
    y_pred = kmeans.predict(X)

    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=60, cmap='Paired')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=60, marker='^',
                c=range(kmeans.n_clusters), linewidths=2, cmap="Paired")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    print("Cluster membership:\n{}".format(y_pred))
    fig.savefig("k_means/complicated_data.png")
