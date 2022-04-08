import logging
import unittest
import pandas as pd

from mlwrap.config import InputData, MLConfig
from mlwrap.enums import DataType, Status
from mlwrap.runners import train
from tests.datasets import IrisDataset


class TestTrain(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        logging.getLogger().setLevel(logging.DEBUG)
        cls.iris = IrisDataset()

    def test_train(self):
        # arrange
        df = pd.concat([self.iris.df_X, self.iris.df_y], axis=1)

        config = MLConfig(
            features=self.iris.features,
            model_feature_id=self.iris.model_feature_id,
            input_data=InputData(data_type=DataType.DataFrame, data_frame=df),
        )

        # act
        result = train(config=config)

        # assert
        self.assertEqual(Status.success, result.status)
