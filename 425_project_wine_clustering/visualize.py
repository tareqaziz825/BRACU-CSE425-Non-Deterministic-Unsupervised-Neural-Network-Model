# handles comparison plots

import matplotlib.pyplot as plt

def plot_clusters(X_train, y_pred, baseline_labels, gmm_labels, som_labels):
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    axs[0,0].scatter(X_train[:, 0], X_train[:, 1], c=y_pred, cmap='viridis')
    axs[0,0].set_title("Custom Model (Latent + KMeans)")

    axs[0,1].scatter(X_train[:, 0], X_train[:, 1], c=baseline_labels, cmap='viridis')
    axs[0,1].set_title("KMeans (Raw Features)")

    axs[1,0].scatter(X_train[:, 0], X_train[:, 1], c=gmm_labels, cmap='viridis')
    axs[1,0].set_title("Gaussian Mixture (Raw Features)")

    axs[1,1].scatter(X_train[:, 0], X_train[:, 1], c=som_labels, cmap='viridis')
    axs[1,1].set_title("SOM Clustering")

    plt.suptitle("Clustering Comparison Across Models", fontsize=14)
    plt.show()
