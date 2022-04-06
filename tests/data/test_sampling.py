import unittest

import pandas as pd
from mlwrap.data.sampling import resample_data

from mlwrap.dto import Feature, MLSettings
from mlwrap.enums import FeatureType, ProblemType

class TestSampling(unittest.TestCase):
    def test_resample_data(self):
        # arrange
        data = pd.DataFrame(
            {
                # Red - 6, Blue - 10  = 13
                "Colour": ["Red", "Blue", "Red", "Blue", "Blue", "Blue", "Blue", "Blue", "Red", "Blue", "Blue", "Blue", "Blue", "Red", "Red", "Red"],
                "Temperature": [30, 15, 14, 28, 30, 15, 16, 29, 27, 33, 14, 15, 32, 32, 28, 29],
                "Weather": ["Sun", "Sun", "Sun", "Sun", "Snow", "Snow", "Sun", "Snow", "Sun", "Snow", "Snow", "Snow", "Sun", "Sun", "Snow", "Sun"],
            })

        features = [
            Feature(id="Colour", feature_type=FeatureType.Categorical),
            Feature(id="Temperature", feature_type=FeatureType.Continuous),
            Feature(id="Weather", feature_type=FeatureType.Categorical)
        ]

        settings = MLSettings(
            model_feature_id="Colour",
            features=features,
            balance_data_via_resampling=True)

        print("Initial 'Red' value counts -",
              data["Colour"].value_counts()["Red"])
        print("Initial 'Blue' value counts -",
              data["Colour"].value_counts()["Blue"])

        # act
        resampled_data = resample_data(data, settings, ProblemType.Classification)

        # assert
        self.assertEqual(10, resampled_data["Colour"].value_counts()["Red"])
        self.assertEqual(10, resampled_data["Colour"].value_counts()["Blue"])