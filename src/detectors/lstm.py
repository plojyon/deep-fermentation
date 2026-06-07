import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from src.sample import get_sample


class _SequenceDataset(Dataset):
    """Dataset for variable-length 2D sequences."""

    def __init__(self, samples: list[np.ndarray], labels: list[int]):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = np.asarray(self.samples[index], dtype=np.float32)
        if sample.size == 0:
            sample = np.zeros((1, 1), dtype=np.float32)
        label = np.float32(self.labels[index])
        return torch.from_numpy(sample), torch.tensor(label, dtype=torch.float32)


def _collate_sequences(batch):
    """Pad a batch of time-major sequences to a common length."""
    samples, labels = zip(*batch)
    sequences = [sample.transpose(0, 1) for sample in samples]
    lengths = torch.tensor([sequence.size(0) for sequence in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    labels = torch.stack(labels)
    return padded, labels, lengths


class LSTMClassifier(nn.Module):
    """Inner LSTM model."""

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        features = hidden[-1]
        features = self.dropout(features)
        logits = self.classifier(features)
        return logits.squeeze(-1)


class LSTMDetector:
    """A bubble detector using an LSTM."""

    display_name: str = "LSTM"

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.25,
        max_sequence_length: int = 10240000,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 20,
        device: str | None = None,
        normalize: bool = True,
        balance_classes: bool = True,
    ):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.normalize = normalize
        self.balance_classes = balance_classes
        self.input_size: int | None = None
        self.model: LSTMClassifier | None = None

    def _build_model(self, input_size: int):
        self.input_size = input_size
        self.model = LSTMClassifier(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
        ).to(self.device)

    def _normalize(self, samples: np.ndarray) -> np.ndarray:
        if self.normalize:
            return (samples - self._mean) / self._std
        return samples

    def _prepare_sample(self, sample: np.ndarray) -> np.ndarray:
        sample = np.asarray(sample, dtype=np.float32)
        if sample.size == 0:
            return np.zeros((1, 1), dtype=np.float32)

        if sample.ndim == 1:
            sample = sample[np.newaxis, :]
        elif sample.ndim > 2:
            sample = sample.reshape(sample.shape[0], -1)

        if sample.shape[-1] > self.max_sequence_length:
            x_old = np.linspace(0.0, 1.0, num=sample.shape[-1], endpoint=True)
            x_new = np.linspace(0.0, 1.0, num=self.max_sequence_length, endpoint=True)
            sample = np.stack(
                [np.interp(x_new, x_old, row).astype(np.float32) for row in sample],
                axis=0,
            )
        return sample

    def train(self, data, positive_intervals, negative_intervals):
        """Train the classifier."""
        pos = []
        neg = []

        print(f"Processing sample of shape {data.shape} for LSTM training.")
        for interval in positive_intervals:
            pos.append(self._prepare_sample(get_sample(data, interval, dimensions=2)))
        for interval in negative_intervals:
            neg.append(self._prepare_sample(get_sample(data, interval, dimensions=2)))

        print(f"Collected {len(pos)} positive and {len(neg)} negative samples for LSTM training.")
        X_train = pos + neg
        y_train = [1] * len(pos) + [0] * len(neg)

        if not X_train:
            raise ValueError("No training samples were collected for the LSTM detector.")

        feature_size = X_train[0].shape[0]
        if self.model is None or self.input_size != feature_size:
            self._build_model(feature_size)

        concatenated = np.concatenate([sample.astype(np.float32).reshape(-1) for sample in X_train])
        self._mean = float(concatenated.mean())
        self._std = float(concatenated.std())

        normalized_samples = [self._normalize(sample) for sample in X_train]
        dataset = _SequenceDataset(normalized_samples, y_train)
        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, collate_fn=_collate_sequences
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        pos_weight = None
        if self.balance_classes and len(pos) > 0 and len(neg) > 0:
            pos_weight_value = len(neg) / max(len(pos), 1)
            pos_weight = torch.tensor([pos_weight_value], device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        assert self.model is not None
        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_inputs, batch_targets, lengths in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)
                lengths = lengths.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch_inputs, lengths)
                loss = criterion(logits, batch_targets)
                loss.backward()
                optimizer.step()

                epoch_loss += float(loss.item()) * batch_inputs.size(0)
                predictions = (torch.sigmoid(logits) >= 0.5).float()
                correct += int((predictions == batch_targets).sum().item())
                total += batch_inputs.size(0)

            print(
                f"Epoch {epoch + 1}/{self.epochs}: "
                f"loss={epoch_loss / max(total, 1):.4f}, "
                f"accuracy={correct / max(total, 1):.3f}"
            )

        print("LSTM training completed.")

    def detect(self, data, intervals):
        """Detect if the sample contains a bubble."""
        predictions = []
        if self.model is None:
            raise ValueError("LSTM detector has not been trained yet.")

        self.model.eval()
        for interval in intervals:
            sample = self._prepare_sample(get_sample(data, interval, dimensions=2))
            sample = self._normalize(sample)
            sample_tensor = torch.from_numpy(sample).transpose(0, 1).unsqueeze(0).to(self.device)
            lengths = torch.tensor([sample_tensor.size(1)], device=self.device)

            with torch.no_grad():
                logit = self.model(sample_tensor, lengths)
                prediction = torch.sigmoid(logit).item() >= 0.5

            predictions.append(bool(prediction))
        return predictions
