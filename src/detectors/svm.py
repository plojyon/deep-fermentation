import joblib
import numpy as np
from sklearn import svm

from src.sample import get_sample


class SVMDetector:
    """A bubble detector using SVM."""

    display_name: str = "SVM"

    def __init__(self):
        self.model = svm.SVC(probability=True)

    def train(self, data, positive_intervals, negative_intervals):
        """Train the classifier."""
        pos = []
        neg = []

        print(f"Processing sample of shape {data.shape} for SVM training.")
        for interval in positive_intervals:
            # pos.append(data[:, interval.start:interval.end])
            # print(f"pos shape: {data[:, interval.start:interval.end].shape}, {interval.start}-{interval.end} = {interval.end-interval.start}")
            pos.append(get_sample(data, interval))
            # print(f"pos shape: {data[:, interval.start].shape}")
        for interval in negative_intervals:
            # neg.append(data[:, interval.start:interval.end])
            # print(f"neg shape: {data[:, interval.start:interval.end].shape}, {interval.start}-{interval.end} = {interval.end-interval.start}")
            neg.append(get_sample(data, interval))
            # print(f"neg shape: {data[:, interval.start].shape}")

        print(f"Collected {len(pos)} positive and {len(neg)} negative samples for SVM training.")
        X_train = np.array(pos + neg)
        y_train = np.array([1] * len(pos) + [0] * len(neg))

        self.model.fit(X_train, y_train)
        print("SVM training completed.")

    def detect(self, data, intervals):
        """Detect if the sample contains a bubble."""
        # prediction = self.model.predict([sample[interval.start:interval.end]])[0]
        predictions = []
        for interval in intervals:
            prediction = self.model.predict([get_sample(data, interval)])
            predictions.append(bool(prediction[0]))
        return predictions
