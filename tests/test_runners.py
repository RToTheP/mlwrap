import unittest

from sklearn.datasets import load_iris

from mlwrap.config import MLConfig
from mlwrap.runners import train


class TestTrain(unittest.TestCase):
    def test_train(self):
        df = load_iris(as_frame=True)
        config = MLConfig(data_frame=df)
        result = train(config=config)
        #self.assertIsNotNone(result)
