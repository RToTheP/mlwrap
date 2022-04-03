import numpy as np
import unittest

from mlwrap.data.encoders import CyclicalEncoder


class TestCyclicalEncoder(unittest.TestCase):
    def test_inverse_transform(self):
        # arrange
        cyclical_period = 7
        data = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        encoder = CyclicalEncoder(cyclical_period)

        # act
        transformed_data = encoder.transform(data)
        inverse_transformed_data = encoder.inverse_transform(transformed_data)

        # assert
        for n in range(data.shape[0]):
            input_row = data[n]
            output_row = inverse_transformed_data[n]
            self.assertAlmostEqual(input_row[0], output_row[0], 5)
