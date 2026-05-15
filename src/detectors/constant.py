import numpy as np
from sklearn import svm


class ConstantDetector:
    """A bubble detector that always returns True."""
    display_name: str = "Constant"

    def train(self, data, positive_intervals, negative_intervals):
        """Train the classifier."""
        pass

    def detect(self, data, intervals):
        """Detect if the sample contains a bubble."""
        return [True for _ in intervals]
