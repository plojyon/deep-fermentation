import numpy as np


class IdentityPreprocessor:
    """A preprocessor that does nothing."""
    display_name: str = "Identity"

    def transform(self, sample):
        """Transform the sample."""
        return np.expand_dims(sample, axis=0)

    def transform_interval(self, interval):
        """Transform a specific interval of the sample."""
        return interval
