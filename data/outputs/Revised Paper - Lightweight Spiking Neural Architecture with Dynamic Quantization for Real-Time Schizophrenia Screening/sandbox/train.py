import torch
from torch.utils.data import DataLoader, TensorDataset
from model import Simplemodel


def main():
    # Synthetic placeholder data — replace with actual dataset loader
    X = torch.randn(256, 128)
    y = torch.randint(0, 10, (256,))
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=32, shuffle=True)

    model = Simplemodel()
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        total_loss = 0.0
        for xb, yb in dl:
            logits = model(xb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/10 — Loss: {total_loss / len(dl):.4f}")
    print("Training complete.")


if __name__ == "__main__":
    main()
