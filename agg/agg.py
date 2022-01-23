import mglearn
from matplotlib import pyplot as plt
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
