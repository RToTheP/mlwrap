from collections import Counter
import time
import logging
from typing import Tuple, Type, List

import numpy as np
import pandas as pd

from mlwrap.enums import Status, FeatureType, HandleUnknown, CleaningType
from mlwrap.config import CleaningReport, CleaningRecord, MLConfig


def clean_training_data(
    data: pd.DataFrame, config: MLConfig
) -> Tuple[Status, pd.DataFrame, CleaningReport]:
    status = Status.success if data is not None else Status.failed

    status, data, cleaning_report = clean_rows(data, config)

    if status != Status.success:
        return

    status, data, regroup_labels_report = regroup_labels(data, config)
    cleaning_report.merge(regroup_labels_report)

    if status != Status.success:
        return status, cleaning_report, data

    status, data, clean_features_report = clean_features(data, config)
    cleaning_report.merge(clean_features_report)

    return status, data, cleaning_report


def clean_inference_data(
    data: pd.DataFrame, config: MLConfig
) -> Tuple[Status, pd.DataFrame, CleaningReport]:
    status = Status.success if data is not None else Status.failed
    status, data, cleaning_report = clean_rows(data, config)
    return status, data, cleaning_report


def clean_rows(
    data: pd.DataFrame, config: MLConfig
) -> Tuple[Status, pd.DataFrame, CleaningReport]:
    t_start = time.perf_counter()
    if data is None:
        return Status.failed, None, None

    status = Status.success
    cleaning_report = CleaningReport()
    initial_row_count = data.shape[0]
    # loop over the rows and clean values
    for feature in config.features.values():
        if feature.id not in data.columns:
            continue

        # if a default value has been provided for a feature then substitute any blanks
        # otherwise, remove the bad rows
        if feature.default_value is not None:
            default_value = (
                feature.default_value
                if feature.feature_type == FeatureType.Categorical
                else float(feature.default_value)
            )
            if (
                data[feature.id].dtype.name == "category"
                and default_value not in data[feature.id].cat.categories
            ):
                data[feature.id].cat.add_categories(default_value, inplace=True)
            data[feature.id].fillna(default_value, inplace=True)
        if feature.feature_type == FeatureType.Continuous:
            lower = float(feature.min_value) if feature.min_value is not None else None
            upper = float(feature.max_value) if feature.max_value is not None else None
            data[feature.id].clip(lower=lower, upper=upper, inplace=True)

        # check the rows with out-of-range values
        # for labels, this is only relevant for inference, hence model_labels should be provided
        if (
            feature.feature_type == FeatureType.Categorical
            and feature.model_labels is not None
            and feature.allowed_labels is not None
        ):
            # Try to replace the values with the "OTHER" label.
            # labels that are outside of the model but within the allowed labels (which is anything that
            # was in the training set irrespective of whether it was removed/regrouped)
            model_labels_ = [x for x in feature.model_labels]
            allowed_labels_ = [x for x in feature.allowed_labels]
            rows_with_knowns = np.bitwise_and(
                ~data[feature.id].isin(model_labels_),
                data[feature.id].isin(allowed_labels_),
            )

            # change the values to OTHER. Note that we don't log cleaning records for these as they are ok from the
            # user's perspective
            data[feature.id] = np.where(
                rows_with_knowns, feature.other_label, data[feature.id]
            )

            # now look at unknowns
            rows_with_unknowns = ~data[feature.id].isin(allowed_labels_)
            cleaning_records_ = [
                CleaningRecord(
                    row=x,
                    label=data[feature.id][x],
                    feature=feature.id,
                    cleaning_type=CleaningType.row_feature_out_of_range,
                )
                for x in data[rows_with_unknowns].index
            ]
            cleaning_report.merge_cleaning_records(cleaning_records_)

            if feature.handle_unknown == HandleUnknown.allow:
                # change the values to OTHER
                data[feature.id] = np.where(
                    rows_with_unknowns, feature.other_label, data[feature.id]
                )

            elif feature.handle_unknown == HandleUnknown.remove:
                # the backstop approach is to drop completely unknown rows
                data.drop(data.loc[rows_with_unknowns].index, inplace=True)

        elif feature.feature_type == FeatureType.Continuous:
            if feature.model_min_value is not None:
                lower = float(feature.model_min_value)
                if feature.min_value is not None:
                    lower = min(lower, float(feature.min_value))
                rows_below = data[feature.id] < lower
                cleaning_records_ = [
                    CleaningRecord(
                        row=x,
                        value=data[feature.id][x],
                        feature=feature.id,
                        cleaning_type=CleaningType.row_feature_out_of_range,
                    )
                    for x in data[rows_below].index
                ]
                cleaning_report.merge_cleaning_records(cleaning_records_)

                if feature.handle_unknown == HandleUnknown.allow:
                    data[feature.id][rows_below] = lower

                elif feature.handle_unknown == HandleUnknown.remove:
                    data.drop(data.loc[rows_below].index, inplace=True)

            if feature.model_max_value is not None:
                upper = float(feature.model_max_value)
                if feature.min_value is not None:
                    upper = max(upper, float(feature.min_value))
                rows_above = data[feature.id] > upper
                cleaning_records_ = [
                    CleaningRecord(
                        row=x,
                        value=data[feature.id][x],
                        feature=feature.id,
                        cleaning_type=CleaningType.row_feature_out_of_range,
                    )
                    for x in data[rows_above].index
                ]
                cleaning_report.merge_cleaning_records(cleaning_records_)

                if feature.handle_unknown == HandleUnknown.allow:
                    data[feature.id][rows_above] = upper

                elif feature.handle_unknown == HandleUnknown.remove:
                    data.drop(data.loc[rows_above].index, inplace=True)

        # finally, check for NaNs and drop/replace them
        start_indices = set(data.index)
        isna_ = data[feature.id].isna()
        count_nonzero_ = np.count_nonzero(isna_)
        if count_nonzero_:
            if feature.handle_unknown == HandleUnknown.remove:
                row_indices_with_nan = data.loc[isna_].index
                data.drop(row_indices_with_nan, inplace=True)
            elif feature.handle_unknown == HandleUnknown.allow:
                replacement = None
                if feature.feature_type == FeatureType.Categorical:
                    # for nan labels, set to the "OTHER" label
                    replacement = feature.other_label
                elif feature.feature_type == FeatureType.Continuous:
                    average_ = None
                    if (
                        feature.model_min_value is not None
                        and feature.model_max_value is not None
                    ):
                        lower = float(feature.model_min_value)
                        upper = float(feature.model_max_value)
                        average_ = 0.5 * (lower + upper)
                    else:
                        average_ = data[feature.id].mean()
                    replacement = average_

                data[feature.id] = np.where(isna_, replacement, data[feature.id])

        invalid_row_ids = start_indices - set(data.index)

        cleaning_records_ = [
            CleaningRecord(
                row=x,
                feature=feature.id,
                cleaning_type=CleaningType.row_feature_out_of_range,
            )
            for x in invalid_row_ids
        ]
        cleaning_report.merge_cleaning_records(cleaning_records_)

    logging.debug(
        f"clean_rows: removed {initial_row_count - data.shape[0]} rows in {time.perf_counter() - t_start:0.4f} seconds"
    )

    if data.shape[0] == 0:
        status = Status.no_valid_rows
        data = None

    return status, data, cleaning_report


def regroup_labels(
    data: pd.DataFrame, config: MLConfig
) -> Tuple[Status, pd.DataFrame, CleaningReport]:
    status = Status.success if data is not None else Status.failed
    cleaning_report = CleaningReport()
    for feature in config.features.values():
        if feature.feature_type != FeatureType.Categorical:
            continue
        if feature.id not in data.columns:
            continue

        value_counts_ = pd.value_counts(data[feature.id])
        value_counts_.sort_values(ascending=False, inplace=True)
        n_labels = np.size(value_counts_)

        # remap labels above the threshold (i.e. towards the tail of the distribution):
        # - evaluate the percentage- or counts-based cutoff
        # - derive boolean masks
        # - if the actual label count is above the threshold, apply the boolean mask to regroup into the OTHER label
        mask_percentage = (value_counts_ / value_counts_.sum() * 100).lt(
            feature.label_percentage_threshold
        )
        count_from_percentage_cut = n_labels - np.count_nonzero(mask_percentage)

        count_from_keep_n_cut = min(feature.keep_n_labels, n_labels)
        nlargest_ = value_counts_.nlargest(n=count_from_keep_n_cut)
        mask_keep_n = ~value_counts_.isin(nlargest_)

        n_labels_to_keep = None
        mask = None
        if count_from_percentage_cut > count_from_keep_n_cut:
            mask = mask_percentage
            n_labels_to_keep = count_from_percentage_cut
        else:
            mask = mask_keep_n
            n_labels_to_keep = count_from_keep_n_cut

        if n_labels - 1 > n_labels_to_keep:  # reserve one for "OTHER" label
            labels_to_regroup = value_counts_[mask].index
            data[feature.id] = np.where(
                data[feature.id].isin(labels_to_regroup),
                feature.other_label,
                data[feature.id],
            )

            if np.size(labels_to_regroup) > 0:
                cleaning_records_ = [
                    CleaningRecord(
                        label=x,
                        feature=feature.id,
                        cleaning_type=CleaningType.label_regrouped,
                    )
                    for x in labels_to_regroup
                ]
                cleaning_report.merge_cleaning_records(cleaning_records_)

            logging.debug(
                f"regroup_labels: feature: {feature.id}, initial: {n_labels}, cpn = {count_from_percentage_cut}, "
                + f"ckn = {count_from_keep_n_cut}, keep: {n_labels_to_keep},"
                + f" after: {np.size(pd.value_counts(data[feature.id]))}"
            )

        # remap labels with a single count
        # only do this if there are more than one though as otherwise
        # the OTHER label could be cleared out
        value_counts_ = pd.value_counts(data[feature.id])
        value_counts_.sort_values(ascending=False, inplace=True)
        n_labels = np.size(value_counts_)

        mask = value_counts_ == 1
        labels_to_regroup = value_counts_[mask].index
        if np.size(labels_to_regroup) > 1:
            rows_to_regroup = data[feature.id].isin(labels_to_regroup)
            # log the change - do this before we make the change otherwise the label will just be OTHER
            cleaning_records_ = [
                CleaningRecord(
                    row=x,
                    label=data[feature.id][x],
                    feature=feature.id,
                    cleaning_type=CleaningType.label_regrouped,
                )
                for x in data[rows_to_regroup].index
            ]
            cleaning_report.merge_cleaning_records(cleaning_records_)

            data[feature.id] = np.where(
                rows_to_regroup, feature.other_label, data[feature.id]
            )

            for label in labels_to_regroup:
                logging.debug(
                    f"regroup single count label {label} on feature: {feature.id}"
                )
    return status, data, cleaning_report


def clean_features(data, config: MLConfig):
    status = Status.success if data is not None else Status.failed
    cleaning_report = CleaningReport()
    while True and status == Status.success:
        restart_cleaning = False
        for feature in config.features.values():
            drop_feature: bool = False

            if feature.id not in data.columns:
                continue

            if feature.feature_type == FeatureType.Categorical:
                value_counts_ = pd.value_counts(data[feature.id])
                n_labels = np.size(value_counts_)
                mask = value_counts_ == 1
                labels_to_cut = value_counts_[mask].index
                if np.size(labels_to_cut) > 0:
                    rows_to_cut = data[feature.id].isin(labels_to_cut)
                    data.drop(data.loc[rows_to_cut].index, inplace=True)

                    cleaning_records_ = [
                        CleaningRecord(
                            label=x,
                            feature=feature.id,
                            cleaning_type=CleaningType.label_counts_too_low,
                        )
                        for x in labels_to_cut
                    ]
                    cleaning_report.merge_cleaning_records(cleaning_records_)

                    for label in labels_to_cut:
                        logging.info(
                            f"Filtering feature {feature.id}, label: {label}, count: {value_counts_[label]}"
                        )

                    restart_cleaning = True
                    logging.debug(f"feature {feature.id} triggering restart_cleaning")

                n_labels = len(data[feature.id].unique())
                drop_feature = n_labels == 1
            elif feature.feature_type == FeatureType.Continuous:
                min_ = data[feature.id].min()
                max_ = data[feature.id].max()

                drop_feature = min_ == max_

            if drop_feature:
                logging.info(f"Dropping feature: {feature.id}")
                feature.active = False

                cleaning_records_ = [
                    CleaningRecord(
                        feature=feature.id,
                        cleaning_type=CleaningType.feature_non_predictive,
                    )
                ]
                cleaning_report.merge_cleaning_records(cleaning_records_)

                data.drop(feature.id, axis=1, inplace=True)
                if feature.id == config.model_feature_id:
                    status = Status.model_feature_removed
                    restart_cleaning = False
                    break

        if not restart_cleaning:
            break

    # check if we have sufficient columns left to train - we need at least 2, i.e. target and training
    if status == Status.success and data is not None and len(data.columns) < 2:
        status = Status.model_feature_count_too_low

    if status != Status.success:
        data = None

    return status, data, cleaning_report


def clean_from_text(text: str, list_of_strings: List[Type[str]]) -> str:
    # Remove each string in list_of_strings from text
    for el in list_of_strings:
        text = text.replace(el, "")
    return text
