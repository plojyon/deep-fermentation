from sklearn.metrics import precision_recall_fscore_support
import joblib

from src.detectors.constant import ConstantDetector
from src.detectors.random import RandomDetector
from src.detectors.random_forest import RandomForest
from src.detectors.svm import SVMDetector
from src.detectors.cnn import CNNDetector
from src.preprocessors.identity import IdentityPreprocessor
from src.preprocessors.stft import StftPreprocessor


class BubbleDetector:
    """A bubble detector."""

    detectors = {
        "constant": ConstantDetector,
        "random": RandomDetector,
        "svm": SVMDetector,
        "random_forest": RandomForest,
        "cnn": CNNDetector,
    }
    preprocessors = {
        "identity": IdentityPreprocessor,
        "stft": StftPreprocessor,
    }

    def __init__(
        self,
        model_name: str,
        preprocessor_name: str,
        model_parameters: dict = {},
        preprocessor_parameters: dict = {},
    ):
        self.model = self.detectors[model_name](**model_parameters)
        self.preprocessor = self.preprocessors[preprocessor_name](**preprocessor_parameters)

        str_prepr_params = "" if not preprocessor_parameters else f"({preprocessor_parameters})"
        str_model_params = "" if not model_parameters else f"({model_parameters})"
        self.name = f"{preprocessor_name}{str_prepr_params}:{model_name}{str_model_params}"

    def train(self, data, positive_intervals, negative_intervals):
        """Train the bubble detector using positive and negative samples."""
        transformed = self.preprocessor.transform(data)
        transformed_positive = [
            self.preprocessor.transform_interval(interval) for interval in positive_intervals
        ]
        transformed_negative = [
            self.preprocessor.transform_interval(interval) for interval in negative_intervals
        ]
        return self.model.train(transformed, transformed_positive, transformed_negative)

    def detect(self, transformed_data, transformed_intervals):
        """Detect bubbles in the STFT representation."""
        return self.model.detect(transformed_data, transformed_intervals)

    def save(self, path: str):
        """Save the trained model to a file."""
        joblib.dump(self, path)

    def evaluate(self, data, positive_intervals, negative_intervals, to_stdout=True):
        """Evaluate the bubble detector on the test set."""
        labels = []
        predictions = []

        # print("data")
        # print(data)
        transformed = self.preprocessor.transform(data)
        # print("transformed")
        # print(transformed)

        labels = [1] * len(positive_intervals) + [0] * len(negative_intervals)
        transformed_intervals = [
            self.preprocessor.transform_interval(interval)
            for interval in positive_intervals + negative_intervals
        ]
        predictions = self.detect(transformed, transformed_intervals)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average="binary", zero_division=0
        )

        if to_stdout:
            print(f"{'='*50}")
            print(self.name)
            print(f"{'='*50}")
            print(
                f"Precision: {precision:.3f} ({precision*100:.0f}% of detections were real bubbles)"
            )
            print(f"Recall:    {recall:.3f} ({recall*100:.0f}% of actual bubbles were detected)")
            print(f"F1-Score:  {f1:.3f}")
            print(f"{'='*50}")
        return precision, recall, f1
