# handles model training with early stopping and logging

import torch
import torch.optim as optim
from loss import loss_function

def train_model(model, X_train, epochs, batch_size, learning_rate, patience):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    early_stopping_counter = 0

    log_file = open("training_log.txt", "w")

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(X_train.shape[0])
        epoch_loss = 0
        for i in range(0, X_train.shape[0], batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = torch.tensor(X_train[indices], dtype=torch.float32)

            x_reconstructed, z, mu, logvar = model(batch_x)
            loss = loss_function(x_reconstructed, batch_x, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            log_file.write(f"Epoch [{epoch+1}/{epochs}], Batch {i//batch_size+1}, Loss: {loss.item():.4f}\n")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= patience:
            print("Early stopping triggered!")
            break

    log_file.close()
    return model
