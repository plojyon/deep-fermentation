import joblib
import numpy as np
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from sample import get_sample


class RandomForest:
    display_name: str = "Random forest"

    def __init__(self, n_estimators=100, random_state=42, max_depth=10):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state, max_depth=max_depth
        )

    def train(self, data, positive_intervals, negative_intervals):
        """Train the classifier."""
        pos = []
        neg = []

        print(f"Processing sample of shape {data.shape} for SVM training.")
        for interval in positive_intervals:
            pos.append(get_sample(data, interval))
        for interval in negative_intervals:
            neg.append(get_sample(data, interval))

        X_train = np.array(pos + neg)
        y_train = np.array([1] * len(pos) + [0] * len(neg))
        self.model.fit(X_train, y_train)

    def detect(self, data, intervals):
        """Detect if the sample contains a bubble."""
        predictions = []
        for interval in intervals:
            prediction = self.model.predict([get_sample(data, interval)])
            predictions.append(bool(prediction[0]))
        return predictions

    def save(self, path: str):
        """Save the trained model to a file."""
        joblib.dump(self.model, path)