import numpy as np
from sklearn.cluster import KMeans
x = np.array([(0.0, 0.0),(9.9, 8.1),(-1.0, 1.0),(7.1, 5.6),(-5.0, -5.5),(8.0, 9.8),(0.5, 0.5)])

cluster_analyzer = KMeans(n_clusters=3, init='k-means++')
cluster_analyzer.fit()
cluster_analyzer.fit(x)
KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
       n_clusters=3, n_init=10, n_jobs=None, precompute_distances='auto',
       random_state=None, tol=0.0001, verbose=0)

cluster_analyzer.cluster_centers_array([[ 8.33333333,  7.83333333],[-0.16666667,  0.5       ],[-5.        , -5.5       ]])

cluster_analyzer.labels_