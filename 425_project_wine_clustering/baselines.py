# KMeans, GMM, and SOM baselines

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from som import SOM

def baseline_models(X_train, y_train, input_dim):
    results = {}

    # KMeans
    kmeans = KMeans(n_clusters=3, random_state=42).fit(X_train)
    kmeans_labels = kmeans.labels_
    results['KMeans'] = (
        silhouette_score(X_train, kmeans_labels),
        adjusted_rand_score(y_train, kmeans_labels),
        normalized_mutual_info_score(y_train, kmeans_labels)
    )

    # GMM
    gmm = GaussianMixture(n_components=3, random_state=42).fit(X_train)
    gmm_labels = gmm.predict(X_train)
    results['GMM'] = (
        silhouette_score(X_train, gmm_labels),
        adjusted_rand_score(y_train, gmm_labels),
        normalized_mutual_info_score(y_train, gmm_labels)
    )

    # SOM
    som = SOM(grid_size=(10, 10), input_dim=input_dim, learning_rate=0.1, n_iterations=1000)
    som.train(X_train)
    som_labels = som.cluster(X_train)
    results['SOM'] = (
        silhouette_score(X_train, som_labels),
        adjusted_rand_score(y_train, som_labels),
        normalized_mutual_info_score(y_train, som_labels)
    )

    return results
