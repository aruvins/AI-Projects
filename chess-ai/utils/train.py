import torch
import numpy as np

from .model import ChessNet
from .synthetic_data import generate_dataset

def train():
    X, y = generate_dataset(15000)
    X = torch.tensor(np.array(X))
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)

    model = ChessNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(5):
        total_loss = 0

        for i in range(0, len(X), 64):
            xb = X[i:i+64]
            yb = y[i:i+64]

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: {total_loss:.4f}")

    torch.save(model.state_dict(), "model/chess_net.pth")


if __name__ == "__main__":
    train()