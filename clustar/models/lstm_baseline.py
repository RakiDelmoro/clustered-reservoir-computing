"""
LSTM Baseline for Moving MNIST Action Classification.
Standard recurrent neural network for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class LSTMBaseline(nn.Module):
    """
    1-layer LSTM + linear classifier.
    Trained with gradient descent (Adam), not ridge regression.
    """

    def __init__(
        self,
        input_dim: int = 128,  # encoded frame dimension
        hidden_dim: int = 128,
        num_classes: int = 4,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM backbone
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Classifier head
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_dim] encoded frames

        Returns:
            logits: [B, num_classes]
        """
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)  # lstm_out: [B, T, hidden]

        # Use last hidden state
        last_hidden = h_n[-1]  # [B, hidden_dim] (last layer)

        # Classify
        logits = self.classifier(last_hidden)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class LSTMWrapper:
    """
    Wrapper for training and evaluating LSTM baseline.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        self.device = device
        self.model = LSTMBaseline(input_dim, hidden_dim, num_classes).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, dataloader, encoder) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            frames = batch["frames"].to(self.device)  # [B, T, 1, H, W]
            labels = batch["label"].to(self.device)

            B, T = frames.shape[0], frames.shape[1]
            # Encode frames using fixed encoder (like CluSTAR)
            with torch.no_grad():
                flat = frames.view(B * T, 1, 64, 64)
                encoded = encoder(flat).view(B, T, -1)  # [B, T, D]

            self.optimizer.zero_grad()
            logits = self.model(encoded)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        return total_loss / len(dataloader), accuracy

    @torch.no_grad()
    def evaluate(self, dataloader, encoder) -> Tuple[float, float]:
        """Evaluate on dataloader."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in dataloader:
            frames = batch["frames"].to(self.device)
            labels = batch["label"].to(self.device)

            B, T = frames.shape[0], frames.shape[1]
            with torch.no_grad():
                flat = frames.view(B * T, 1, 64, 64)
                encoded = encoder(flat).view(B, T, -1)

            logits = self.model(encoded)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        return total_loss / len(dataloader), accuracy


def test_lstm():
    """Quick test."""
    from models.encoder import SpatialEncoder

    device = "cpu"
    encoder = SpatialEncoder(output_dim=128).to(device)
    wrapper = LSTMWrapper(input_dim=128, hidden_dim=128, num_classes=4, device=device)

    # Dummy data
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 20

        def __getitem__(self, idx):
            return {
                "frames": torch.rand(30, 1, 64, 64),
                "label": torch.randint(0, 4, ()).long(),
            }

    loader = DataLoader(DummyDataset(), batch_size=4)

    # Train one epoch
    loss, acc = wrapper.train_epoch(loader, encoder)
    print(f"Train loss: {loss:.4f}, acc: {acc:.2%}")

    # Eval
    loss, acc = wrapper.evaluate(loader, encoder)
    print(f"Eval loss: {loss:.4f}, acc: {acc:.2%}")

    print("LSTM test passed!")


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    test_lstm()
