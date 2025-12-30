import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from model import HiringModel
from utils import compute_fairness_metrics, get_shap_values, get_lime_explanation


def generate_placeholder_data(num_samples: int = 1000):
    """Create synthetic data mimicking the UCI Adult structure.

    Returns:
        X (np.ndarray): Features, last column is a binary protected attribute.
        y (np.ndarray): Binary target.
    """
    np.random.seed(0)
    # 10 numeric features
    X_numeric = np.random.randn(num_samples, 10)
    # Binary protected attribute (e.g., gender)
    protected = np.random.binomial(1, 0.5, size=(num_samples, 1))
    X = np.concatenate([X_numeric, protected], axis=1).astype(np.float32)
    # Simple linear rule for label with slight bias against protected group
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.2 * protected[:, 0] > 0).astype(np.float32)
    return X, y


def main():
    X, y = generate_placeholder_data()
    input_dim = X.shape[1]

    # Split into train / test
    X_train, X_test = X[:800], X[800:]
    y_train, y_test = y[:800], y[800:]

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = HiringModel(input_dim)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb).squeeze()
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    # Evaluation
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.from_numpy(X_test)).squeeze().numpy()
    protected_attr = X_test[:, -1]
    dp, eo = compute_fairness_metrics(test_preds, y_test, protected_attr)
    print(f"Demographic Parity Difference: {dp:.4f}")
    print(f"Equal Opportunity Difference: {eo:.4f}")

    # Explanations for a single test instance
    instance = X_test[:1]
    shap_vals = get_shap_values(model, X_train, instance)
    lime_exp = get_lime_explanation(model, X_train, instance)
    print("SHAP values for the instance:", shap_vals)
    print("LIME explanation for the instance:", lime_exp)


if __name__ == "__main__":
    main()
