import matplotlib.pyplot as plt
import mglearn
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler


def execute():
    X, y = make_moons(n_samples=200, noise=0.05, random_state=0)

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3), subplot_kw={'xticks': (), 'yticks': ()})
    algorithms = [KMeans(n_clusters=2), AgglomerativeClustering(n_clusters=2), DBSCAN()]

    random_state = np.random.RandomState(seed=0)
    random_clusters = random_state.randint(low=0, high=2, size=len(X))

    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=random_clusters, cmap=mglearn.cm3, s=60)
    axes[0].set_title("Random assignment - ARI: {:.2f}".format(adjusted_rand_score(y,random_clusters)))

    for ax,algorithm in zip(axes[1:],algorithms):
        clusters = algorithm.fit_predict(X_scaled)
        ax.scatter(X_scaled[:,0],X_scaled[:,1],c=clusters,cmap=mglearn.cm3,s=60)
        ax.set_title("{} - ARI: {:.2f}".format(algorithm.__class__.__name__,adjusted_rand_score(y,clusters)))

    fig.savefig('assessment/ari.png')
