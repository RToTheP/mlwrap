import os
import numpy as np
import pandas as pd
import unittest
import logging


from mlwrap.config import MLConfig, Feature, LabelDetail
from mlwrap.data.cleaning import clean_features, clean_rows, regroup_labels

from mlwrap.enums import Status, FeatureType, HandleUnknown, CleaningType


class TestCleaning(unittest.TestCase):
    @ classmethod
    def setUpClass(cls):
        logging.getLogger().setLevel(logging.DEBUG)

    def test_clean_features_and_labels_no_data(self):
        # arrange
        data = None
        config = MLConfig()

        # act
        status = clean_features(data, config)[0]

        # assert
        self.assertEqual(Status.failed, status)

    def test_clean_features_and_labels_success(self):
        # arrange
        data = pd.DataFrame(
            {
                "Colour": ["Red", "Blue", "Red", "Blue"],
                "Temperature": ["Hot", "Cold", "Cold", "Hot"]
            })
        features = [
            Feature(id="Colour", feature_type=FeatureType.Categorical),
            Feature(id="Temperature", feature_type=FeatureType.Categorical)
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status = clean_features(data, config)[0]

        # assert
        self.assertEqual(Status.success, status)

    def test_clean_features_model_feature_removed(self):
        # arrange
        data = pd.DataFrame(
            {
                "Colour": ["Red", "Blue", "Red"],
                "Temperature": ["Hot", "Cold", "Cold"]
            })
        features = [
            Feature(id="Colour", feature_type=FeatureType.Categorical),
            Feature(id="Temperature", feature_type=FeatureType.Categorical)
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status = clean_features(data, config)[0]

        # assert
        self.assertEqual(Status.model_feature_removed, status)

    def test_clean_features_string_feature_removed(self):
        # arrange
        data = pd.DataFrame(
            {
                "Colour": ["Red", "Blue", "Red", "Blue", "Blue"],
                "Temperature": ["Hot", "Cold", "Cold", "Hot", "Hot"],
                "Weather": ["Sun", "Sun", "Sun", "Sun", "Snow"]
            })
        features = [
            Feature(id="Colour", feature_type=FeatureType.Categorical),
            Feature(id="Temperature", feature_type=FeatureType.Categorical),
            Feature(id="Weather", feature_type=FeatureType.Categorical)
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status, _, cleaning_report = clean_features(
            data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertTrue(
            any(x.feature == "Weather" and x.cleaning_type == CleaningType.feature_non_predictive
                for x in cleaning_report.cleaning_records))
        self.assertTrue(any(x.label == "Snow" and x.feature ==
                            "Weather" for x in cleaning_report.cleaning_records))

    def test_clean_features_double_feature_removed(self):
        # arrange
        data = pd.DataFrame(
            {
                "Colour": [0, 1, 2, 0, 1],
                "Temperature": [0, 1, 2, 0, 1],
                "Weather": [0, 0, 0, 0, 0]
            })
        features = [
            Feature(id="Colour", feature_type=FeatureType.Continuous),
            Feature(id="Temperature", feature_type=FeatureType.Continuous),
            Feature(id="Weather", feature_type=FeatureType.Continuous)
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status, _, cleaning_report = clean_features(
            data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertTrue(
            any(x.feature == "Weather" and x.cleaning_type == CleaningType.feature_non_predictive
                for x in cleaning_report.cleaning_records))

    def test_clean_features_label_removed(self):
        # arrange
        data = pd.DataFrame(
            {
                "Colour": ["Red", "Blue", "Red", "Blue", "Blue", "Red"],
                "Temperature": ["Hot", "Cold", "Cold", "Hot", "Hot", "Mild"]
            })
        features = [
            Feature(id="Colour", feature_type=FeatureType.Categorical),
            Feature(id="Temperature", feature_type=FeatureType.Categorical)
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status, _, cleaning_report = clean_features(
            data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertFalse(any(
            x.cleaning_type == CleaningType.feature_non_predictive for x in cleaning_report.cleaning_records))
        self.assertEqual(1, len([x for x in cleaning_report.cleaning_records
                                 if x.label == "Mild" and x.feature == "Temperature"]))


    def test_clean_rows_remove_no_valid_double_rows(self):
        # arrange
        data = pd.DataFrame(
            {
                "Height": [np.nan],
                "Temperature": [99]
            })
        features = [
            Feature(
                id="Height",
                feature_type=FeatureType.Continuous,
                handle_unknown=HandleUnknown.remove),
            Feature(id="Temperature", feature_type=FeatureType.Continuous)
        ]
        config = MLConfig(
            model_feature_id="Height", features=features)

        # act
        status, _, cleaning_report = clean_rows(
            data, config)

        # assert
        self.assertEqual(Status.no_valid_rows, status)
        self.assertEqual(1, len([x for x in cleaning_report.cleaning_records
                                 if x.cleaning_type == CleaningType.row_feature_out_of_range and x.feature == "Height"]))

    def test_clean_rows_allow_nan_double_rows(self):
        # arrange
        data = pd.DataFrame(
            {
                "Height": [np.nan, 10],
                "Temperature": [99, 60]
            })
        features = [
            Feature(
                id="Height",
                feature_type=FeatureType.Continuous,
                handle_unknown=HandleUnknown.allow
            ),
            Feature(id="Temperature", feature_type=FeatureType.Continuous)
        ]
        config = MLConfig(
            model_feature_id="Height", features=features)

        # act
        status, _, cleaning_report = clean_rows(
            data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(0, len(
            [x for x in cleaning_report.cleaning_records if x.cleaning_type == CleaningType.row_feature_out_of_range]))
        self.assertEqual(1, len(data.Height.unique()))
        self.assertEqual(10, data.Height.unique()[0])

    def test_clean_rows_remove_nan_double_rows(self):
        # arrange
        data = pd.DataFrame(
            {
                "Height": [np.nan, 10],
                "Temperature": [99, 60]
            })
        features = [
            Feature(
                id="Height",
                feature_type=FeatureType.Continuous,
                handle_unknown=HandleUnknown.remove
            ),
            Feature(id="Temperature", feature_type=FeatureType.Continuous)
        ]
        config = MLConfig(
            model_feature_id="Height", features=features)

        # act
        status, _, cleaning_report = clean_rows(
            data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(1, len([x for x in cleaning_report.cleaning_records
                                 if x.feature == "Height" and x.cleaning_type == CleaningType.row_feature_out_of_range]))

    def test_clean_rows_remove_nan_string_rows(self):
        # arrange
        data = pd.DataFrame(
            {
                "Colour": [np.nan],
                "Temperature": ["Hot"]
            })
        features = [
            Feature(
                id="Colour",
                feature_type=FeatureType.Categorical,
                handle_unknown=HandleUnknown.remove),
            Feature(id="Temperature", feature_type=FeatureType.Categorical)
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status, _, cleaning_report = clean_rows(
            data, config)

        # assert
        self.assertEqual(Status.no_valid_rows, status)
        self.assertEqual(1, len([x for x in cleaning_report.cleaning_records
                                 if x.feature == "Colour" and x.cleaning_type == CleaningType.row_feature_out_of_range]))

    def test_clean_rows_allow_nan_string_rows(self):
        # arrange
        data = pd.DataFrame(
            {
                "Colour": [np.nan],
                "Temperature": ["Hot"]
            })
        features = [
            Feature(
                id="Colour",
                feature_type=FeatureType.Categorical,
                handle_unknown=HandleUnknown.allow,
                other_label="OTHER"),
            Feature(id="Temperature", feature_type=FeatureType.Categorical)
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status, _, cleaning_report = clean_rows(
            data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(0, len([x for x in cleaning_report.cleaning_records
                                 if x.cleaning_type == CleaningType.row_feature_out_of_range]))
        self.assertEqual("OTHER", data.Colour.unique()[0])

    def test_clean_rows_remove_some_bad_rows(self):
        # arrange
        data = pd.DataFrame(
            {
                "Colour": [np.nan, "Blue", "Red", "Blue", "Blue", "Red"],
                "Temperature": ["Hot", np.nan, "Cold", "Hot", "Hot", "Mild"]
            })
        features = [
            Feature(
                id="Colour",
                feature_type=FeatureType.Categorical,
                handle_unknown=HandleUnknown.remove),
            Feature(
                id="Temperature",
                feature_type=FeatureType.Categorical,
                handle_unknown=HandleUnknown.remove)
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status, _, cleaning_report = clean_rows(
            data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(2, len([x.cleaning_type ==
                                 CleaningType.row_feature_out_of_range for x in cleaning_report.cleaning_records]))
        self.assertEqual(1, len([x for x in cleaning_report.cleaning_records
                                 if x.cleaning_type == CleaningType.row_feature_out_of_range and
                                 x.feature == "Colour"]))
        self.assertEqual(1, len([x for x in cleaning_report.cleaning_records
                                 if x.cleaning_type == CleaningType.row_feature_out_of_range and
                                 x.feature == "Temperature"]))

    def test_clean_rows_allow_some_bad_rows(self):
        # arrange
        data = pd.DataFrame(
            {
                "Colour": [np.nan, "Blue", "Red", "Blue", "Blue", "Red"],
                "Temperature": ["Hot", np.nan, "Cold", "Hot", "Hot", "Mild"]
            })
        features = [
            Feature(
                id="Colour",
                feature_type=FeatureType.Categorical,
                handle_unknown=HandleUnknown.allow,
                other_label="OTHER"),
            Feature(
                id="Temperature",
                feature_type=FeatureType.Categorical,
                handle_unknown=HandleUnknown.allow,
                other_label="OTHER")
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status, data, cleaning_report = clean_rows(
            data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(0, len([x for x in cleaning_report.cleaning_records
                                 if x.cleaning_type == CleaningType.row_feature_out_of_range]))
        self.assertTrue("OTHER" in data.Colour.unique())
        self.assertTrue("OTHER" in data.Temperature.unique())

    def test_clean_rows_default_values(self):
        # arrange
        data = pd.DataFrame(
            {
                "Colour": [np.nan, "Blue", "Red", "Blue", "Blue", "Red"],
                "Temperature": ["Hot", np.nan, "Cold", "Hot", "Hot", "Mild"]
            })
        features = [
            Feature(id="Colour", feature_type=FeatureType.Categorical,
                          default_value="Yellow"),
            Feature(id="Temperature", feature_type=FeatureType.Categorical,
                          default_value="Tepid")
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status, _, cleaning_report = clean_rows(data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(0, len([x for x in cleaning_report.cleaning_records
                                 if x.cleaning_type == CleaningType.row_feature_out_of_range]))

    def test_clean_rows_default_values_numbers(self):
        # arrange
        data = pd.DataFrame(
            {
                "Number": [np.nan, 2, 3, 4, 5, 6]
            })
        features = [
            Feature(id="Number", feature_type=FeatureType.Continuous,
                          default_value="1")
        ]
        config = MLConfig(
            model_feature_id="Number", features=features)

        # act
        status, _, cleaning_report = clean_rows(data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(0, len([x for x in cleaning_report.cleaning_records
                                 if x.cleaning_type == CleaningType.row_feature_out_of_range]))

    def test_clean_rows_min_max_values(self):
        # arrange
        data = pd.DataFrame(
            {
                "Number": [-10, 2, 3, 4, 5, 10000]
            })
        features = [
            Feature(id="Number", feature_type=FeatureType.Continuous,
                          min_value="0", max_value=6)
        ]
        config = MLConfig(
            model_feature_id="Number", features=features)

        # act
        status, data, _ = clean_rows(data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(float(features[0].min_value), data.min()[0])
        self.assertEqual(features[0].max_value, data.max()[0])

    def test_clean_rows_remove_out_of_range_label(self):
        # arrange
        data = pd.DataFrame(
            {
                "Colour": ["Red", "Yellow", "Red", "Blue", "Green", "Red"],
                "Temperature": ["Hot", "Cold", "Cold", "Hot", "Hot", "Mild"]
            })
        features = [
            Feature(id="Colour", feature_type=FeatureType.Categorical,
                          model_labels=[LabelDetail(x)
                                        for x in ["Red", "Blue"]],
                          allowed_labels=[LabelDetail(x)
                                          for x in ["Red", "Blue"]],
                          handle_unknown=HandleUnknown.remove),
            Feature(id="Temperature", feature_type=FeatureType.Categorical)
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status, _, cleaning_report = clean_rows(data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(2, len([x for x in cleaning_report.cleaning_records
                                 if x.cleaning_type == CleaningType.row_feature_out_of_range and
                                 x.row in [1, 4]]))

    def test_clean_rows_remove_out_of_range_double(self):
        # arrange
        data = pd.DataFrame(
            {
                "Number": [1, 2, -1, 4, 5, 6]
            })
        features = [
            Feature(id="Number",
                          feature_type=FeatureType.Continuous,
                          model_min_value="1",
                          handle_unknown=HandleUnknown.remove)
        ]
        config = MLConfig(
            model_feature_id="Number", features=features)

        # act
        status, _, cleaning_report = clean_rows(data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(1, len(cleaning_report.cleaning_records))
        self.assertEqual(2, cleaning_report.cleaning_records[0].row)

    def test_clean_rows_remove_out_of_range_label_with_indexed_rows(self):
        # arrange
        data = pd.DataFrame(
            {
                "Colour": ["Red", "Yellow", "Red", "Blue", "Green", "Red"],
                "Temperature": ["Hot", "Cold", "Cold", "Hot", "Hot", "Mild"]
            },
            index=["zero", "one", "two", "three", "four", "five"])
        features = [
            Feature(id="Colour", feature_type=FeatureType.Categorical,
                          model_labels=[LabelDetail(x)
                                        for x in ["Red", "Blue"]],
                          allowed_labels=[LabelDetail(x)
                                          for x in ["Red", "Blue"]],
                          handle_unknown=HandleUnknown.remove),
            Feature(id="Temperature", feature_type=FeatureType.Categorical)
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status, _, cleaning_report = clean_rows(data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(2, len(
            [x for x in cleaning_report.cleaning_records
             if x.cleaning_type == CleaningType.row_feature_out_of_range and x.row in ["one", "four"]]))

    def test_clean_rows_remove_out_of_range_double_with_indexed_rows(self):
        # arrange
        data = pd.DataFrame(
            {
                "Number": [1, 2, -1, 4, 5, 6]
            },
            index=["zero", "one", "two", "three", "four", "five"])
        features = [
            Feature(id="Number",
                          feature_type=FeatureType.Continuous,
                          model_min_value="1",
                          handle_unknown=HandleUnknown.remove)
        ]
        config = MLConfig(
            model_feature_id="Number", features=features)

        # act
        status, _, cleaning_report = clean_rows(data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(1, len(cleaning_report.cleaning_records))
        self.assertEqual("two", cleaning_report.cleaning_records[0].row)

    def test_clean_rows_allow_out_of_range_label(self):
        # arrange
        data = pd.DataFrame(
            {
                "Colour": ["Red", "Yellow", "Red", "Blue", "Green", "Red"],
                "Temperature": ["Hot", "Cold", "Cold", "Hot", "Hot", "Mild"]
            })
        features = [
            Feature(id="Colour", feature_type=FeatureType.Categorical,
                          model_labels=[LabelDetail(x)
                                        for x in ["Red", "Blue"]],
                          allowed_labels=[LabelDetail(x)
                                          for x in ["Red", "Blue"]],
                          handle_unknown=HandleUnknown.allow,
                          other_label="OTHER"),
            Feature(id="Temperature", feature_type=FeatureType.Categorical)
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status, data, cleaning_report = clean_rows(data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(2, len(
            [x for x in cleaning_report.cleaning_records
             if x.cleaning_type == CleaningType.row_feature_out_of_range
             and x.label in ["Yellow", "Green"]]))
        self.assertTrue("OTHER" in data.Colour.unique())

    def test_clean_rows_allow_out_of_range_label_with_alloweds(self):
        # arrange
        data = pd.DataFrame(
            {
                "Colour": ["Red", "Yellow", "Red", "Blue", "Green", "Red"],
                "Temperature": ["Hot", "Cold", "Cold", "Hot", "Hot", "Mild"]
            })
        features = [
            Feature(id="Colour", feature_type=FeatureType.Categorical,
                          model_labels=[LabelDetail(x)
                                        for x in ["Red", "Blue"]],
                          allowed_labels=[LabelDetail(x) for x in ["Red", "Blue",
                                                                   "Yellow", "Green", "OTHER"]],
                          handle_unknown=HandleUnknown.allow,
                          other_label="OTHER"),
            Feature(id="Temperature", feature_type=FeatureType.Categorical)
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status, data, cleaning_report = clean_rows(data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(0, len(cleaning_report.cleaning_records))
        self.assertTrue("OTHER" in data.Colour.unique())

    def test_clean_rows_allow_out_of_range_double(self):
        # arrange
        data = pd.DataFrame(
            {
                "Number": [1, 2, -1, 4, 5, 6]
            })

        features = [
            Feature(id="Number",
                          feature_type=FeatureType.Continuous,
                          model_min_value="1",
                          handle_unknown=HandleUnknown.allow)
        ]
        config = MLConfig(
            model_feature_id="Number", features=features)

        # act
        status, data, cleaning_report = clean_rows(data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(1,
                         len([x for x in cleaning_report.cleaning_records
                              if x.cleaning_type == CleaningType.row_feature_out_of_range
                              and x.value == -1]))
        self.assertFalse(any(x == -1 for x in data.Number))

    def test_clean_rows_allow_out_of_range_label_with_indexed_rows(self):
        # arrange
        data = pd.DataFrame(
            {
                "Colour": ["Red", "Yellow", "Red", "Blue", "Green", "Red"],
                "Temperature": ["Hot", "Cold", "Cold", "Hot", "Hot", "Mild"]
            },
            index=["zero", "one", "two", "three", "four", "five"])
        features = [
            Feature(id="Colour", feature_type=FeatureType.Categorical,
                          model_labels=[LabelDetail(x)
                                        for x in ["Red", "Blue"]],
                          allowed_labels=[LabelDetail(x)
                                          for x in ["Red", "Blue"]],
                          handle_unknown=HandleUnknown.allow,
                          other_label="OTHER"),
            Feature(id="Temperature", feature_type=FeatureType.Categorical)
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status, data, cleaning_report = clean_rows(data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(2, len([x for x in cleaning_report.cleaning_records
                                 if x.cleaning_type == CleaningType.row_feature_out_of_range
                                 and x.label in ["Yellow", "Green"]]))
        self.assertTrue("OTHER" in data.Colour.unique())

    def test_clean_rows_allow_out_of_range_double_with_indexed_rows(self):
        # arrange
        data = pd.DataFrame(
            {
                "Number": [1, 2, -1, 4, 5, 6]
            },
            index=["zero", "one", "two", "three", "four", "five"])
        features = [
            Feature(id="Number",
                          feature_type=FeatureType.Continuous,
                          model_min_value="1",
                          handle_unknown=HandleUnknown.allow)
        ]
        config = MLConfig(
            model_feature_id="Number", features=features)

        # act
        status, data, cleaning_report = clean_rows(data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(1, len([x for x in cleaning_report.cleaning_records
                                 if x.cleaning_type == CleaningType.row_feature_out_of_range
                                 and x.value == -1]))
        self.assertFalse(any(x == -1 for x in data.Number))


    def test_regroup_labels_by_count_keep_one(self):
        # arrange
        colours = np.concatenate((
            np.repeat("Red", 10),
            np.repeat("Yellow", 1),
            np.repeat("Blue", 1),
            np.repeat("Green", 1)))
        data = pd.DataFrame(
            {
                "Colour": colours
            })
        features = [
            Feature(
                id="Colour",
                feature_type=FeatureType.Categorical,
                keep_n_labels=4,
                other_label="OTHER")
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status, data, cleaning_report = regroup_labels(
            data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(3, len([x.feature == "Colour" and x.cleaning_type == CleaningType.label_regrouped
                                 for x in cleaning_report.cleaning_records]))
        self.assertTrue(all(x in data.Colour.unique()
                            for x in ["Red", "OTHER"]))

    def test_regroup_labels_by_count_keep_all(self):
        # arrange
        colours = np.concatenate((
            np.repeat("Red", 10),
            np.repeat("Yellow", 2),
            np.repeat("Blue", 2),
            np.repeat("Green", 2)))
        data = pd.DataFrame(
            {
                "Colour": colours
            })
        features = [
            Feature(
                id="Colour", feature_type=FeatureType.Categorical, keep_n_labels=4)
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status, data, cleaning_report = regroup_labels(
            data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(0, len(
            [x.cleaning_type == CleaningType.label_regrouped for x in cleaning_report.cleaning_records]))
        self.assertTrue(all(x in data.Colour.unique()
                            for x in ["Red", "Yellow", "Blue", "Green"]))

    """Label regrouping test where the tail of the distribution is flat so although keep_n_labels is 2, we
    end up keeping all 4 labels"""

    def test_regroup_labels_by_count_flat_tail(self):
        # arrange
        colours = np.concatenate((
            np.repeat("Red", 10),
            np.repeat("Yellow", 2),
            np.repeat("Blue", 2),
            np.repeat("Green", 2)))
        data = pd.DataFrame(
            {
                "Colour": colours
            })
        features = [
            Feature(
                id="Colour", feature_type=FeatureType.Categorical, keep_n_labels=2)
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status, data, cleaning_report = regroup_labels(
            data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertEqual(0, len(
            [x.cleaning_type == CleaningType.label_regrouped for x in cleaning_report.cleaning_records]))
        self.assertEqual(4, len(data.Colour.unique()))

    def test_regroup_labels_by_count_keep_fewer(self):
        # arrange
        colours = np.concatenate((
            np.repeat("Red", 10),
            np.repeat("Yellow", 5),
            np.repeat("Blue", 1),
            np.repeat("Green", 1)))
        data = pd.DataFrame(
            {
                "Colour": colours
            })
        features = [
            Feature(
                id="Colour", feature_type=FeatureType.Categorical, keep_n_labels=2)
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status, data, cleaning_report = regroup_labels(
            data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertTrue(all(
            x.cleaning_type == CleaningType.label_regrouped and x.label in [
                "Blue", "Green"]
            for x in cleaning_report.cleaning_records))
        self.assertEqual(3, len(data.Colour.unique()))

    def test_regroup_labels_by_percentage_keep_fewer(self):
        # arrange
        colours = np.concatenate((
            np.repeat("Red", 10),
            np.repeat("Yellow", 5),
            np.repeat("Blue", 1),
            np.repeat("Green", 1)))
        data = pd.DataFrame(
            {
                "Colour": colours
            })
        features = [
            Feature(
                id="Colour",
                feature_type=FeatureType.Categorical,
                keep_n_labels=0,
                label_percentage_threshold=10)
        ]
        config = MLConfig(
            model_feature_id="Colour", features=features)

        # act
        status, data, cleaning_report = regroup_labels(
            data, config)

        # assert
        self.assertEqual(Status.success, status)
        self.assertTrue(
            all(x.label in ["Blue", "Green"] and x.cleaning_type == CleaningType.label_regrouped
                for x in cleaning_report.cleaning_records))
        self.assertEqual(3, len(data.Colour.unique()))
