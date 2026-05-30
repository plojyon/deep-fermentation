import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.sample import get_sample


class CNNClassifier2D(nn.Module):
    """Inner CNN model."""

    def __init__(
        self,
        conv_channels: tuple[int, ...],
        kernel_size: int | tuple[int, int],
        pool_kernel: tuple[int, int],
        hidden_dim: int,
        dropout: float,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        padding = tuple(size // 2 for size in kernel_size)
        layers: list[nn.Module] = []
        in_channels = 1
        for out_channels in conv_channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=pool_kernel, stride=pool_kernel),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x.squeeze(-1)


class CNNDetector:
    """A bubble detector using a deep CNN."""

    display_name: str = "CNN (2D)"

    def __init__(
        self,
        conv_channels: tuple[int, ...] = (16, 32, 64),
        kernel_size: int | tuple[int, int] = 3,
        pool_kernel: tuple[int, int] = (1, 2),
        hidden_dim: int = 64,
        dropout: float = 0.25,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 32,
        epochs: int = 20,
        device: str | None = None,
        seed: int = 42,
        normalize: bool = True,
        balance_classes: bool = True,
    ):
        self.conv_channels = tuple(conv_channels)
        self.kernel_size = kernel_size
        self.pool_kernel = pool_kernel
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.seed = seed
        self.normalize = normalize
        self.balance_classes = balance_classes

        self.model = CNNClassifier2D(
            conv_channels=self.conv_channels,
            kernel_size=self.kernel_size,
            pool_kernel=self.pool_kernel,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)

        self.mean_: float | None = None
        self.std_: float | None = None

    def _prepare_training_data(self, samples: list[np.ndarray]) -> np.ndarray:
        return np.stack([np.asarray(sample, dtype=np.float32) for sample in samples], axis=0)

    def _normalize(self, samples: np.ndarray) -> np.ndarray:
        if not self.normalize:
            return samples

        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("CNNDetector normalization parameters are not initialized")

        return (samples - self.mean_) / self.std_

    def _prepare_sample_tensor(self, sample: np.ndarray) -> torch.Tensor:
        sample = np.asarray(sample, dtype=np.float32)
        if self.normalize:
            if self.mean_ is None or self.std_ is None:
                raise RuntimeError("CNNDetector normalization parameters are not initialized")
            sample = (sample - self.mean_) / self.std_
        return torch.from_numpy(sample).unsqueeze(0)

    def train(self, data, positive_intervals, negative_intervals):
        """Train the classifier."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        pos = []
        neg = []

        print(f"Processing sample of shape {data.shape} for CNN training.")
        for interval in positive_intervals:
            pos.append(get_sample(data, interval, dimensions=2))
        for interval in negative_intervals:
            neg.append(get_sample(data, interval, dimensions=2))

        print(f"Collected {len(pos)} positive and {len(neg)} negative samples for CNN training.")
        X_train = self._prepare_training_data(pos + neg)
        y_train = np.array([1] * len(pos) + [0] * len(neg))

        if self.normalize:
            self.mean_ = float(X_train.mean())
            self.std_ = float(X_train.std())
            if self.std_ == 0:
                self.std_ = 1.0
            X_train = self._normalize(X_train)

        print(
            "Training CNN. Data shape:",
            X_train.shape,
            "Labels shape:",
            y_train.shape,
            "Sanity check: 2 =",
            X_train.ndim,
        )

        inputs = torch.from_numpy(X_train).unsqueeze(1)
        targets = torch.from_numpy(y_train.astype(np.float32))
        dataset = TensorDataset(inputs, targets)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        pos_weight = None
        if self.balance_classes and len(pos) > 0 and len(neg) > 0:
            pos_weight_value = len(neg) / max(len(pos), 1)
            pos_weight = torch.tensor([pos_weight_value], device=self.device)

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.model.train()
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(self.device)
                batch_targets = batch_targets.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(batch_inputs)
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

        print("CNN training completed.")

    def detect(self, data, intervals):
        """Detect if the sample contains a bubble."""
        predictions = []
        self.model.eval()
        expected_shape = None
        for interval in intervals:
            sample_array = get_sample(data, interval, dimensions=2)
            if expected_shape is None:
                expected_shape = sample_array.shape
            elif sample_array.shape != expected_shape:
                # Edge case: last interval may be shorter and will break the convolution
                predictions.append(False)
                continue

            sample = self._prepare_sample_tensor(sample_array)
            sample = sample.unsqueeze(0).to(self.device)

            with torch.no_grad():
                logit = self.model(sample)
                prediction = torch.sigmoid(logit).item() >= 0.5

            predictions.append(bool(prediction))
        return predictions
