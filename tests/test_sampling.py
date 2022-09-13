from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, SMOTENC
import pandas as pd
import pytest

from mlwrap import sampling
from mlwrap.config import Feature, MLConfig
from mlwrap.enums import FeatureType, ProblemType
from tests.datasets import IrisDataset


def test_get_resampler_smote(iris: IrisDataset):
    config = MLConfig(
        model_feature_id=iris.model_feature_id,
        problem_type=ProblemType.Classification,
        balance_data_via_resampling=True,
    )
    resampler = sampling.get_resampler(iris.X, config, config.problem_type)
    X, y = resampler.fit_resample(iris.X, iris.y)

    assert isinstance(resampler, SMOTE)
    assert X is not None
    assert y is not None


def test_get_resampler_smotenc():
    df_X = pd.DataFrame(
        data={
            "foo": [
                "A",
                "B",
                "A",
                "A",
                "B",
                "A",
                "A",
                "B",
                "A",
                "A",
                "B",
                "A",
                "A",
                "A",
                "B",
                "A",
            ],
            "bar": [0, 1, 2, 0, 1, 2, 2, 0, 1, 1, 1, 0, 1, 1, 1, 0],
            "car": [
                "C",
                "D",
                "C",
                "D",
                "C",
                "D",
                "D",
                "C",
                "D",
                "C",
                "C",
                "C",
                "D",
                "C",
                "C",
                "C",
            ],
        }
    )
    df_X["foo"] = df_X.foo.astype("category")
    df_X["car"] = df_X.car.astype("category")
    df_y = df_X.pop("car")
    config = MLConfig(
        model_feature_id="car",
        problem_type=ProblemType.Classification,
        balance_data_via_resampling=True,
    )
    resampler = sampling.get_resampler(df_X, config, config.problem_type)
    X, y = resampler.fit_resample(df_X, df_y)

    assert isinstance(resampler, SMOTENC)
    assert X is not None
    assert y is not None


def test_resample_data():
    # arrange
    data = pd.DataFrame(
        {
            # Red - 6, Blue - 10  = 13
            "Colour": [
                "Red",
                "Blue",
                "Red",
                "Blue",
                "Blue",
                "Blue",
                "Blue",
                "Blue",
                "Red",
                "Blue",
                "Blue",
                "Blue",
                "Blue",
                "Red",
                "Red",
                "Red",
            ],
            "Temperature": [
                30,
                15,
                14,
                28,
                30,
                15,
                16,
                29,
                27,
                33,
                14,
                15,
                32,
                32,
                28,
                29,
            ],
            "Weather": [
                "Sun",
                "Sun",
                "Sun",
                "Sun",
                "Snow",
                "Snow",
                "Sun",
                "Snow",
                "Sun",
                "Snow",
                "Snow",
                "Snow",
                "Sun",
                "Sun",
                "Snow",
                "Sun",
            ],
        }
    )

    features = [
        Feature(id="Colour", feature_type=FeatureType.Categorical),
        Feature(id="Temperature", feature_type=FeatureType.Continuous),
        Feature(id="Weather", feature_type=FeatureType.Categorical),
    ]

    config = MLConfig(
        model_feature_id="Colour",
        features=features,
        balance_data_via_resampling=True,
    )

    print("Initial 'Red' value counts -", data["Colour"].value_counts()["Red"])
    print("Initial 'Blue' value counts -", data["Colour"].value_counts()["Blue"])

    # act
    X = data
    y = X.pop("Colour")
    resampler = sampling.get_resampler(data, config, ProblemType.Classification)
    X_resampled, y_resampled = resampler.fit_resample(X, y)

    # assert
    value_counts = y_resampled.value_counts()
    assert value_counts["Red"] == 10
    assert value_counts["Blue"] == 10

def test_random_sample(iris: IrisDataset):
    X_sample = sampling.random_sample(iris.X, 100)
    assert X_sample.shape[0] == 100
    assert X_sample.shape[1] == iris.X.shape[1]
