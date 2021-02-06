import matplotlib.pyplot as plt
#from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

features, true_labels = make_blobs(n_samples=200,centers=3,cluster_std=2.75,random_state=42)
features[:5]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features[:5]
kmeans = KMeans(init="random",n_clusters=3,n_init=10,max_iter=300,random_state=42)

kmeans.fit(scaled_features)
kmeans.inertia_
kmeans.cluster_centers_
kmeans.n_iter_
kmeans.labels_[:5]

kmeans_kwargs = {"init": "random","n_init": 10,"max_iter": 300,"random_state": 42,}
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)
silhouette_coefficients = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    score = silhouette_score(scaled_features, kmeans.labels_)
    silhouette_coefficients.append(score)

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score
features, true_labels = make_moons(n_samples=250, noise=0.05, random_state=42)
scaled_features = scaler.fit_transform(features)
kmeans = KMeans(n_clusters=2)
dbscan = DBSCAN(eps=0.3)
kmeans.fit(scaled_features)
dbscan.fit(scaled_features)
kmeans_silhouette = silhouette_score(scaled_features, kmeans.labels_).round(2)
dbscan_silhouette = silhouette_score(scaled_features, dbscan.labels_).round (2)
kmeans_silhouette
dbscan_silhouette

ari_kmeans = adjusted_rand_score(true_labels, kmeans.labels_)
ari_dbscan = adjusted_rand_score(true_labels, dbscan.labels_)
round(ari_kmeans, 2)
round(ari_dbscan, 2)