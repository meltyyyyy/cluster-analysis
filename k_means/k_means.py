import matplotlib.pyplot as plt
import mglearn as mglearn
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs


def execute():
    X, y = make_blobs(random_state=1)

    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)

    print("Cluster membership:\n{}".format(kmeans.labels_))

