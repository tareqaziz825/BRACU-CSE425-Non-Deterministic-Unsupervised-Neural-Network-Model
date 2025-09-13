# python main.py

from config import *
from data import load_wine_data
from model import StochasticClusteringNN
from train import train_model
from evaluation import evaluate_model, quantify_uncertainty
from baselines import baseline_models
from visualize import plot_clusters

X_train, X_test, y_train, y_test = load_wine_data()
input_dim = X_train.shape[1]

# Train custom model
model = StochasticClusteringNN(input_dim, hidden_dim, latent_dim)
model = train_model(model, X_train, epochs, batch_size, learning_rate, early_stopping_patience)

# Evaluate custom model
silhouette, ari, nmi, y_pred = evaluate_model(model, X_train, y_train)
print(f"Custom Model -> Silhouette: {silhouette:.4f}, ARI: {ari:.4f}, NMI: {nmi:.4f}")

# Uncertainty
uncertainty = quantify_uncertainty(model, X_train)
print(f"Uncertainty (avg variance): {uncertainty:.4f}")

# Baselines
baseline_results = baseline_models(X_train, y_train, input_dim)
for method, (sil, ari_b, nmi_b) in baseline_results.items():
    print(f"{method} -> Silhouette: {sil:.4f}, ARI: {ari_b:.4f}, NMI: {nmi_b:.4f}")

# Visualization
plot_clusters(X_train, y_pred,
              baseline_labels=baseline_results['KMeans'][0],
              gmm_labels=baseline_results['GMM'][0],
              som_labels=baseline_results['SOM'][0])
