import unittest

from sklearn.datasets import load_iris

from mlwrap.dto import MLSettings
from mlwrap.runners import train


class TestTrain(unittest.TestCase):
    def test_train(self):
        df = load_iris(as_frame=True)
        settings = MLSettings(data_frame=df)
        result = train(settings=settings)
        self.assertIsNotNone(result)
