import numpy as np


class RandomDetector:
    """A bubble detector that returns a random result."""
    display_name: str = "Random"

    def train(self, data, positive_intervals, negative_intervals):
        """Train the classifier."""
        pass

    def detect(self, data, intervals):
        """Detect if the sample contains a bubble."""
        return [np.random.choice([True, False]) for _ in intervals]
