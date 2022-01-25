import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA


def dbscan(X_pca,X_people,image_shape):
    dbscan = DBSCAN()
    labels = dbscan.fit_predict(X_pca)
    print("Unique labels: {}".format(np.unique(labels)))

    dbscan = DBSCAN(min_samples=3)
    labels = dbscan.fit_predict(X_pca)
    print("Unique labels: {}".format(np.unique(labels)))

    dbscan = DBSCAN(min_samples=3,eps=15)
    labels = dbscan.fit_predict(X_pca)
    print("Unique labels: {}".format(np.unique(labels)))
    print("Number of points per cluster: {}".format(np.bincount(labels + 1)))

    noise = X_people[labels == -1]
    fig,axes = plt.subplots(3,9,subplot_kw={'xticks': (),'yticks': ()},figsize=(12,4))
    for image,ax in zip(noise,axes.ravel()):
        ax.imshow(image.reshape(image_shape),vmin=0,vmax=1)
    fig.savefig('assessment/dbscan.png')

    for eps in [1,3,5,7,9,11,13]:
        print('\neps={}'.format(eps))
        dbscan = DBSCAN(eps=eps,min_samples=3)
        labels = dbscan.fit_predict(X_pca)
        print('Clusters present: {}'.format(np.unique(labels)))
        print('Cluster sizes: {}'.format(np.bincount(labels + 1)))

def execute():
    people = fetch_lfw_people(min_faces_per_person=20, resize=0.7)
    image_shape = people.images[0].shape

    mask = np.zeros(people.target.shape, dtype=bool)
    for target in np.unique(people.target):
        mask[np.where(people.target == target)[0][:50]] = 1

    X_people = people.data[mask]
    y_people = people.target[mask]
    X_people = X_people / 255
    pca = PCA(n_components=100, whiten=True, random_state=0)
    pca.fit(X_people)
    X_pca = pca.transform(X_people)

    dbscan(X_pca,X_people,image_shape)
