import mglearn
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import ward, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs


def execute():
    X, y = make_blobs(random_state=1)

    agg = AgglomerativeClustering(n_clusters=3)
    assignment = agg.fit_predict(X)

    fig = plt.figure()
    mglearn.discrete_scatter(X[:, 0], X[:, 1], assignment)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    fig.savefig("agg/agglomerative_clustering.png")

    X, y = make_blobs(random_state=0, n_samples=12)
    linkage_array = ward(X)

    fig = plt.figure()
    dendrogram(linkage_array)
    ax = plt.gca()
    bounds = ax.get_xbound()
    ax.plot(bounds, [7.25, 7.25], '--', c='k')
    ax.plot(bounds, [4, 4], '--', c='k')
    ax.text(bounds[1], 7.25, 'two clusters', va='center', fontdict={'size': 15})
    ax.text(bounds[1], 4, 'three clusters', va='center', fontdict={'size': 15})
    plt.xlabel('Sample index')
    plt.ylabel('Cluster distance')
    fig.savefig('agg/dendrogram.png')
