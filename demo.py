import torch
import torch.nn as nn
import torch.optim as optim


class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def main() -> None:
    torch.manual_seed(42)

    samples = 500
    x = torch.randn(samples, 1)
    y = 3 * x + 2 + 0.3 * torch.randn(samples, 1)

    net = NeuralNetwork()
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01)

    epochs = 1000
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = net(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 200 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss.item():.6f}")

    test_x = torch.tensor([[4.0], [7.5], [-2.0]])
    test_y = net(test_x).detach()

    print("\nPredictions:")
    for value, prediction in zip(test_x, test_y):
        print(f"x={value.item():>5.1f} -> y={prediction.item():>7.3f}")


if __name__ == "__main__":
    main()
