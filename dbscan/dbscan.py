import matplotlib.pyplot as plt
import mglearn
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler


def execute():
    X,y = make_moons(noise=0.05,random_state=0,n_samples=200)

    scaler = StandardScaler()
    scaler.fit(X)
    X_scaled = scaler.transform(X)

    dbscan = DBSCAN()
    clusters = dbscan.fit_predict(X_scaled)

    fig = plt.figure()
    plt.scatter(X_scaled[:,0],X_scaled[:,1],c=clusters,cmap=mglearn.cm2,s=60)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    fig.savefig("dbscan/dbscan.png")

