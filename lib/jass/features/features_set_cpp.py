from typing import Union

import jasscpp
import numpy as np


class FeaturesSetCpp:
    """
    Calculate features for Jass from PlayerRound. Different features are possible with different details.
    This abstract base class defines the interface for the features.
    """
    # subclasses define a feature length both on class and instance level
    FEATURE_LENGTH = 0                          # type: int

    def __init__(self):
        """
        Initialize
        """
        # the length of the feature vector produced by this feature generator
        self._feature_length = FeaturesSetCpp.FEATURE_LENGTH  # type: int

    @property
    def feature_length(self):
        return self._feature_length

    def convert_to_features(self, obs: Union[jasscpp.GameObservationCpp, jasscpp.GameStateCpp],
                            rule: jasscpp.RuleSchieberCpp) -> np.ndarray:
        """
        Convert the obs to a feature vector.
        """
        raise NotImplementedError()

    def decode_features(self, features: np.ndarray)->dict:
        """
        Decode the features into an object for easier logging and post processing. As the features might not
        cover the complete information of a player round, a complete PlayerRound object can not be reconstructed
        in all cases.
        Args:
            features: the 1D feature vector to convert
        Returns:
            dictionary of the decoded features, the entries depend on the features used
        """
        raise NotImplementedError()