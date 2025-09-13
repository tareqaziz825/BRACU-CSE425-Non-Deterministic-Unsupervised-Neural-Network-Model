# evaluation of the custom model + uncertainty

import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score

def evaluate_model(model, X_train, y_train, n_clusters=3):
    model.eval()
    with torch.no_grad():
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        _, z_train, _, _ = model(X_train_tensor)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(z_train.numpy())
    y_pred = kmeans.labels_

    silhouette = silhouette_score(X_train, y_pred)
    ari = adjusted_rand_score(y_train, y_pred)
    nmi = normalized_mutual_info_score(y_train, y_pred)

    return silhouette, ari, nmi, y_pred

def quantify_uncertainty(model, X_train, num_samples=10, n_clusters=3):
    all_preds = []
    with torch.no_grad():
        for _ in range(num_samples):
            X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
            _, z_train, _, _ = model(X_train_tensor)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(z_train.numpy())
            all_preds.append(kmeans.labels_)
    all_preds = np.array(all_preds)
    return np.mean(np.var(all_preds, axis=0))
