#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File name: data.py
Author: Yannick Heiser
Created: 2025-09-19
Version: 1.0
Description:
    This script reads in data
Contact: yahei@dtu.dk
Dependencies: pandas, os, typing, pathlib, data_validation
"""

from pathlib import Path
import logging
import os
from io import StringIO
from urllib.parse import quote
from omegaconf import DictConfig, OmegaConf
import hydra
import inspect
from typing import Dict, Any
import pandas as pd
from datetime import datetime
import openmeteo_requests
import json
from entsoe import EntsoePandasClient, mappings
import requests
from retry_requests import retry
from day_ahead_v2.utils.data_validation import (
    ValidationReport, ShapeInfo, MissingValuesInfo, MissingColumnInfo,
    DuplicateInfo, MemoryUsageInfo, NumericSummaryStats
)

# Initialize logger
logger = logging.getLogger(__name__)

class DataHandler:
    """
    Initiates empty data handler.

    Attributes:
        _data (pd.DataFrame): Loaded data (DataFrame).
    """

    def __init__(self) -> None:
        """
        Initialize a DataHandler.
        """
        self._data = pd.DataFrame()

    @property
    def data(self) -> pd.DataFrame:
        """
        Access the loaded data.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame for
            supported formats, or an empty DataFrame if no file was loaded.
        """
        return self._data

    def set_data(self, data: pd.DataFrame, filename: str | None = None) -> None:
        """
        Set data directly without loading from file.

        Args:
            data (pd.DataFrame): The data to set.
            filename (str | None): Optional filename for reference.
        """
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
                logger.info("Input data converted to DataFrame.")
            except Exception as e:
                logger.error(f"Failed to convert input data to DataFrame: {e}")
                raise ValueError("Input data must be convertible to a pandas DataFrame.") from e
        self._data: pd.DataFrame = data

    def validate_data(self) -> ValidationReport:
        """
        Validate the loaded DataFrame and return a ValidationReport.
        """
        # Check if data is loaded
        if self._data.empty:
            logger.warning("Empty data. No validation performed.")
            return ValidationReport(
                data_type="DataFrame",
                shape=ShapeInfo(rows=0, columns=0),
                missing_values=MissingValuesInfo(total_missing=0, by_column={}),
                data_types={},
                duplicates=DuplicateInfo(count=0, percentage=0.0),
                memory_usage=MemoryUsageInfo(total_mb=0.0, by_column_kb={}),
                numeric_summary={},
                column_completeness={}
            )

        # Create reference to data
        df = self._data

        # Shape
        shape = ShapeInfo(rows=df.shape[0], columns=df.shape[1])

        # Missing values
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df)) * 100

        missing_values = MissingValuesInfo(
            total_missing=int(missing_counts.sum()),
            by_column={
                col: MissingColumnInfo(
                    count=int(missing_counts[col]),
                    percentage=float(missing_percentages[col])
                )
                for col in missing_counts.index if missing_counts[col] > 0
            }
        )

        # Duplicates
        dup_count = int(df.duplicated().sum())
        duplicates = DuplicateInfo(
            count=dup_count,
            percentage=(dup_count / len(df)) * 100
        )

        # Memory usage
        mem_raw = df.memory_usage(deep=True)
        memory_usage = MemoryUsageInfo(
            total_mb=mem_raw.sum() / 1024 / 1024,
            by_column_kb={
                col: mem_raw[col] / 1024
                for col in mem_raw.index
            }
        )

        # Numeric summary
        numeric_summary = {}
        numeric_cols = df.select_dtypes(include="number").columns
        for col in numeric_cols:
            numeric_summary[col] = NumericSummaryStats(
                count=int(df[col].count()),
                mean=float(df[col].mean()),
                std=float(df[col].std()),
                min=float(df[col].min()),
                q25=float(df[col].quantile(0.25)),
                q50=float(df[col].quantile(0.5)),
                q75=float(df[col].quantile(0.75)),
                max=float(df[col].max()),
                zeros=int((df[col] == 0).sum()),
                negative=int((df[col] < 0).sum())
            )

        # Column completeness
        column_completeness = {
            col: 100 - missing_percentages[col]
            for col in df.columns
        }

        report = ValidationReport(
            data_type="DataFrame",
            shape=shape,
            missing_values=missing_values,
            data_types={col: str(dtype) for col, dtype in df.dtypes.items()},
            duplicates=duplicates,
            memory_usage=memory_usage,
            numeric_summary=numeric_summary,
            column_completeness=column_completeness
        )

        # Log the report
        report.log()

        return report

    def transform_data(self, cfg: DictConfig | None = None) -> "DataHandler":
        """
        Apply data transformations and return a new instance.

        Args:
            cfg (DictConfig | None): Configuration for transformations.

        Returns:
            DataHandler: New instance with transformed data.
        """
        if cfg is None:
            logger.info("No transformation configuration provided. Returning original data.")
            return self

        # Validate input data
        if self._data.empty:
            logger.error("No data loaded. Transformation aborted.")
            return self

        if not isinstance(self._data, pd.DataFrame):
            logger.warning("Transformations only supported for pandas DataFrames.")
            return self

        df = self._data.copy()

        # Set index to datetime column if specified
        datetime_column = cfg.datasets.training.get("datetime_column", None)

        if datetime_column is not None and datetime_column in df.columns:
            df.set_index(datetime_column, inplace=True)
            logger.info("Set index to datetime column: %s", datetime_column)

        log: list[str] = []

        self._remove_classes(df, cfg.datasets, log)
        self._label_mapping(df, cfg.datasets, log)
        self._drop_columns(df, cfg.datasets, log)
        self._rename_columns(df, cfg.datasets, log)
        self._drop_duplicates(df, cfg.datasets, log)
        self._drop_missing_columns(df, cfg.datasets, log)
        self._fill_missing(df, cfg.datasets, log)
        self._convert_types(df, cfg.datasets, log)
        self._normalize_columns(df, cfg.datasets, log)
        self._hot_encode(df, cfg.datasets, log)
        self._to_dtype_float32(df)

        return self._build_new_handler(df, log)

    def _remove_classes(self, df: pd.DataFrame, cfg: DictConfig, log: list[str]) -> None:
        """
        Remove specified classes from the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to modify in-place
            cfg (DictConfig): Transformation config section
            log (list[str]): Transformation log
        """
        remove_classes = cfg.transform.get("remove_classes")
        if not remove_classes:
            logger.debug("No remove_classes specified in config.")
            return

        remove_classes_cfg = cfg.transform.remove_classes

        for column, classes_to_remove in remove_classes_cfg.items():
            if column not in df.columns:
                logger.warning(f"remove_classes skipped: column '{column}' not found.")
                continue

            # Remove specified classes
            for class_ in classes_to_remove:
                original_count = df.shape[0]
                if class_ in df[column].values:
                    df.drop(df[df[column] == class_].index, inplace=True)
                    removed_count = original_count - df.shape[0]
                    log.append(f"Removed {removed_count} rows from column '{column}' with class '{class_}'")



    def _label_mapping(self, df: pd.DataFrame, cfg: DictConfig, log: list[str]) -> None:
        """
        Apply label mappings defined in the config to target columns.

        Args:
            df (pd.DataFrame): DataFrame to modify in-place
            cfg (DictConfig): Transformation config section
            log (list[str]): Transformation log
        """

        mapping = cfg.transform.get("label_mapping")
        if not mapping:
            logger.debug("No label mapping specified. Skipping label mapping.")
            return

        label_mapping_cfg = cfg.transform.label_mapping

        for column, mapping in label_mapping_cfg.items():
            if column not in df.columns:
                logger.warning(f"Label mapping skipped: column '{column}' not found.")
                continue

            # Convert OmegaConf -> dict
            mapping_dict = dict(mapping)

            # Keep keys as-is (usually strings), cast values to int
            try:
                mapping_clean = {str(k): int(v) for k, v in mapping_dict.items()}
            except ValueError as e:
                raise ValueError(
                    f"Invalid label mapping for column '{column}'. "
                    f"Values must be castable to int."
                ) from e

            # Work directly on original column (no numeric conversion)
            series = df[column].astype(str)

            # Check for unseen labels
            unique_labels = set(series.dropna().unique())
            expected_labels = set(mapping_clean.keys())
            unseen = unique_labels - expected_labels

            if unseen:
                raise ValueError(
                    f"Unmapped labels found in column '{column}': {sorted(unseen)}. "
                    f"Expected only: {sorted(expected_labels)}"
                )

            # Apply mapping
            df[column] = series.map(mapping_clean).astype("int64")

            msg = (
                f"Applied label mapping to '{column}': "
                f"{mapping_clean}"
            )
            log.append(msg)

    def _drop_columns(self, df: pd.DataFrame, cfg: DictConfig, log: list[str]) -> None:
        """Drop specified columns from the DataFrame."""
        cols = cfg.transform.get("drop_columns")

        if not cols:
            return

        existing = [c for c in cols if c in df.columns]
        missing = [c for c in cols if c not in df.columns]

        if existing:
            df.drop(columns=existing, inplace=True)
            log.append(f"Dropped columns: {existing}")

        if missing:
            log.append(f"Skipped missing columns: {missing}")

    def _rename_columns(self, df: pd.DataFrame, cfg: DictConfig, log: list[str]) -> None:
        rename_dict = cfg.transform.get("rename_columns")

        if not rename_dict:
            return

        existing = {k: v for k, v in rename_dict.items() if k in df.columns}
        missing = [k for k in rename_dict.keys() if k not in df.columns]

        if existing:
            df.rename(columns=existing, inplace=True)
            log.append(f"Renamed columns: {existing}")

        if missing:
            log.append(f"Skipped missing columns for renaming: {missing}")

    def _drop_duplicates(self, df: pd.DataFrame, cfg: DictConfig, log: list[str]) -> None:
        """Drop duplicate rows from the DataFrame."""
        if cfg.transform.get("drop_duplicates", False):
            original_rows = len(df)
            df.drop_duplicates(inplace=True)
            removed = original_rows - len(df)
            log.append(f"Removed {removed} duplicate rows")

    def _drop_missing_columns(self, df: pd.DataFrame, cfg: DictConfig, log: list[str]) -> None:
        """Drop columns with high missing value percentages and remove them from feature_columns if present."""
        threshold = cfg.transform.get("drop_missing_threshold")

        if not threshold:
            return

        if 0.0 <= threshold <= 1.0:
            missing_pct = df.isnull().mean()
            cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
            if cols_to_drop:
                # Drop from DataFrame
                df.drop(columns=cols_to_drop, inplace=True)
                log.append(f"Dropped columns with >{threshold*100}% missing: {cols_to_drop}")

                # Remove from feature_columns if present
                if hasattr(cfg.training, "feature_columns_flex") and cfg.training.feature_columns_flex:
                    original_features = list(cfg.training.feature_columns_flex)
                    cfg.training.feature_columns_flex = [
                        col for col in cfg.training.feature_columns_flex if col not in cols_to_drop
                    ]
                    removed = set(original_features) - set(cfg.training.feature_columns_flex)
                    if removed:
                        log.append(f"Removed from feature_columns_flex in config: {list(removed)}")
                if hasattr(cfg.training, "feature_columns_fix") and cfg.training.feature_columns_fix:
                    original_features = list(cfg.training.feature_columns_fix)
                    cfg.training.feature_columns_fix = [
                        col for col in cfg.training.feature_columns_fix if col not in cols_to_drop
                    ]
                    removed = set(original_features) - set(cfg.training.feature_columns_fix)
                    if removed:
                        log.append(f"Removed from feature_columns_fix in config: {list(removed)}")
            else:
                log.append(f"No columns exceeded missing threshold ({threshold*100}%)")
        else:
            logger.warning("drop_missing_threshold must be between 0.0 and 1.0")

    def _fill_missing(self, df: pd.DataFrame, cfg: DictConfig, log: list[str]) -> None:
        """Fill missing values using specified method."""
        method = cfg.transform.fill_missing.get("method")
        limit = cfg.transform.fill_missing.get("limit", None)

        if not method:
            return

        if method in {"mean", "median", "interpolate"}:
            numeric_cols = df.select_dtypes(include="number").columns
            if numeric_cols.empty:
                log.append(f"Skipped fill_missing ({method}): no numeric columns found")
                return

        if method == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        elif method == "median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        elif method == "mode":
            for col in df.columns:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col] = df[col].fillna(mode_val[0])

        elif method == "forward":
            df.ffill(limit=limit, inplace=True)
            df.bfill(limit=limit, inplace=True)

        elif method == "backward":
            df.bfill(limit=limit, inplace=True)
            df.ffill(limit=limit, inplace=True)

        elif method == "zero":
            df.fillna(0, inplace=True)

        elif method == "interpolate":
            if limit is None:
                df[numeric_cols] = df[numeric_cols].interpolate(limit_direction="both")
            else:
                df[numeric_cols] = df[numeric_cols].interpolate(limit=limit, limit_direction="both")
        else:
            log.append(f"Unknown fill_missing method: {method}")
            return

        suffix = f" (limit={limit})" if limit is not None else ""
        log.append(f"Filled missing values using: {method}{suffix}")

    def _convert_types(self, df: pd.DataFrame, cfg: DictConfig, log: list[str]) -> None:
        """Convert column data types."""
        type_dict = cfg.transform.get("convert_types")

        if not type_dict:
            return

        existing = {col: dtype for col, dtype in type_dict.items() if col in df.columns}
        missing = [col for col in type_dict.keys() if col not in df.columns]

        for col, dtype in existing.items():
            try:
                df[col] = df[col].astype(dtype)
                log.append(f"Converted {col} to {dtype}")
            except Exception as e:
                log.append(f"Failed to convert {col}: {str(e)[:30]}")

        if missing:
            log.append(f"Skipped missing columns for type conversion: {missing}")

    def _normalize_columns(self, df: pd.DataFrame, cfg: DictConfig, log: list[str]) -> None:
        """Normalize specified columns."""
        cols = cfg.transform.get("normalize", {}).get("columns")
        method = cfg.transform.get("normalize", {}).get("method", "zscore")

        if not cols:
            return

        existing = [c for c in cols if c in df.columns]
        missing = [c for c in cols if c not in df.columns]

        for col in existing:
            if pd.api.types.is_numeric_dtype(df[col]):
                if method == "minmax":
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val != min_val:
                        df[col] = (df[col] - min_val) / (max_val - min_val)
                        log.append(f"Normalized {col} using min-max scaling")
                elif method == "zscore":
                    mean_val = df[col].mean()
                    std_val = df[col].std()
                    if std_val != 0:
                        df[col] = (df[col] - mean_val) / std_val
                        log.append(f"Standardized {col} using z-score")
                else:
                    log.append(f"Unknown normalization method: {method} for column {col}")

        if missing:
            log.append(f"Skipped missing columns for normalization: {missing}")

    def _to_dtype_float32(self, df: pd.DataFrame) -> None:
        """
        Convert all numeric and boolean columns to float32 for ML compatibility.
        Logs any remaining columns that are neither numeric nor boolean.
        """
        # Select all numeric or boolean columns
        numeric_bool_cols = df.select_dtypes(include=["number", "bool"]).columns.tolist()

        # Columns actually converted (exclude those already float32)
        to_convert = [col for col in numeric_bool_cols if df[col].dtype != "float32"]

        # Convert
        df[to_convert] = df[to_convert].astype("float32")

        # Columns that are neither numeric nor boolean
        other_cols = [col for col in df.columns if col not in numeric_bool_cols]

        # Log conversion summary
        if other_cols:
            # Include their dtypes
            dtypes_info = {col: str(df[col].dtype) for col in other_cols}
            logger.warning(f"Columns neither numeric nor boolean and not converted: {dtypes_info}")
        else:
            logger.info("All columns are numeric/boolean and converted to float32.")

    def _hot_encode(self, df: pd.DataFrame, cfg: DictConfig, log: list[str]) -> None:
        """
        One-hot encode specified categorical columns.

        Args:
            df (pd.DataFrame): DataFrame to modify in-place
            cfg (DictConfig): Transformation config section
            log (list[str]): Transformation log
        """
        cols = cfg.transform.get("hot_encode", {}).get("columns")

        if not cols:
            return

        existing = [c for c in cols if c in df.columns]
        missing = [c for c in cols if c not in df.columns]

        if not existing:
            log.append("No valid columns found for one-hot encoding.")
            return

        # Perform one-hot encoding
        df_encoded = pd.get_dummies(
            df[existing],
            prefix=existing,
            drop_first=False,
            dtype="float32",
        )

        # Drop original columns
        df.drop(columns=existing, inplace=True)

        # Join encoded columns
        df[df_encoded.columns] = df_encoded

        log.append(f"One-hot encoded columns: {existing}")
        log.append(f"Created columns: {list(df_encoded.columns)}")

        if missing:
            log.append(f"Skipped missing columns for one-hot encoding: {missing}")

        # If one-hot encoded columns are in feature_columns, update the config
        if hasattr(cfg.training, "feature_columns_flex") and cfg.training.feature_columns_flex:
            # Remove original columns
            cfg.training.feature_columns_flex = [
                col for col in cfg.training.feature_columns_flex if col not in existing
            ]
            log.append(f"Updated feature_columns_flex in config: removed {existing}")
        if hasattr(cfg.training, "feature_columns_fix") and cfg.training.feature_columns_fix:
            # Remove original columns
            cfg.training.feature_columns_fix = [
                col for col in cfg.training.feature_columns_fix if col not in existing
            ]
            # Add new one-hot columns
            cfg.training.feature_columns_fix.extend(df_encoded.columns.tolist())
            log.append(f"Updated feature_columns_fix in config: removed {existing}, added {list(df_encoded.columns)}")

    def _build_new_handler(self, df: pd.DataFrame, log: list[str]) -> "DataHandler":
        """Build a new DataHandler instance from a transformed DataFrame."""
        new_handler = DataHandler.__new__(DataHandler)

        # Preserve metadata
        new_handler.filename = (
            f"transformed_{self.filename}"
            if self.filename is not None
            else None
        )
        new_handler.filepath = self.filepath
        new_handler.arguments = self.arguments.copy()

        # Assign transformed data
        new_handler._data = df

        # Log transformation summary
        logger.info("Data transformation completed")
        logger.info("Transformation steps:")
        for entry in log:
            logger.info("  • %s", entry)

        logger.info(
            "Final shape: %s rows, %s columns",
            df.shape[0],
            df.shape[1],
        )

        return new_handler

    def cut_data(self, start: pd.Timestamp, end: pd.Timestamp, datetime_column: str) -> "DataHandler":
        """
        Cut the data to a specific datetime range.

        Args:
            start (pd.Timestamp): Start of the datetime range (inclusive).
            end (pd.Timestamp): End of the datetime range (inclusive).
            datetime_column (str): Name of the datetime column to filter on.

        Raises:
            ValueError: If datetime_column does not exist or is not datetime type.
            ValueError: If no data is loaded.
            ValueError: If datetime column in data is timezone-naive.

        Returns:
            DataHandler: New instance with data cut to the specified range.
        """
        if self._data is None or self._data.empty:
            raise ValueError("No data loaded to cut.")

        datetime_index = False
        if datetime_column not in self._data.columns:
            # check if datetime_column is the index
            if self._data.index.name != datetime_column:
                raise ValueError(f"Column '{datetime_column}' does not exist in data.")
            else:
                datetime_index = True
                self._data = self._data.reset_index()
                logger.info(f"Reset index to access datetime column '{datetime_column}'.")
        else:
            if self._data.index.name != datetime_column:
                datetime_index = True


        if not pd.api.types.is_datetime64_any_dtype(self._data[datetime_column]):
            try:
                self._data[datetime_column] = pd.to_datetime(self._data[datetime_column], utc=True)
                logger.warning(f"Converted column '{datetime_column}' to datetime with UTC timezone.")
            except (ValueError, TypeError):
                raise ValueError(f"Column '{datetime_column}' is not of datetime type and cannot be converted.")

        # Check timezone awareness compatibility
        if (self._data[datetime_column].dt.tz is None):
            raise ValueError("Datetime column in data is timezone-naive, but must be timezone-aware.")

        mask = (self._data[datetime_column] >= start) & (self._data[datetime_column] < end)
        cut_df = self._data.loc[mask].copy()

        if datetime_index:
            self._data.set_index(datetime_column, inplace=True)
            cut_df.set_index(datetime_column, inplace=True)

        logger.info(
            "Data cut to range %s to %s. New shape: %s rows, %s columns",
            start,
            end,
            cut_df.shape[0],
            cut_df.shape[1],
        )

        return self._build_new_handler(cut_df, [f"Cut data to range {start} - {end}"])

    def save(self, savename: str | None = None, save_dir: str | None = None, **kwargs) -> None:
        """
        Save the data to disk. Supports CSV, Parquet, Excel, JSON formats.

        Args:
            savename (str | None): Name of the output file. If None, defaults to
                original savename with '_saved' suffix and '.csv' extension.
            save_dir (str | None): Directory to save the file. If None, uses
                the original file's directory or current working directory.
            **kwargs: Additional arguments passed to the pandas to_* functions.

        Raises:
            ValueError: If no data is loaded.
            ValueError: If filename does not include an extension or has unsupported format.
            ValueError: If saving non-DataFrame data is attempted.
        Returns:
            Path: The path to the saved file.
        """
        if self._data is None or self._data.empty:
            raise ValueError("No data to save. Load data first.")

        # Supported extensions
        writers = {
            ".csv": lambda p: self._data.to_csv(p, **kwargs),
            ".parquet": lambda p: self._data.to_parquet(p, **kwargs),
            ".xlsx": lambda p: self._data.to_excel(p, **kwargs),
            ".xls": lambda p: self._data.to_excel(p, **kwargs),
            ".json": lambda p: self._data.to_json(p, orient="records", indent=2, **kwargs),
        }

        # Resolve filename
        if savename:
            output_name = Path(savename)
        else:
            base = "data"
            output_name = Path(f"{base}_saved.csv")

        ext = output_name.suffix.lower()
        if not ext:
            raise ValueError("Filename must include a file extension")
        if ext not in writers:
            raise ValueError(f"Unsupported file format: {ext}")

        # Resolve directory
        if save_dir:
            save_dir = Path(save_dir)
        else:
            save_dir = Path.cwd()

        if not save_dir.is_absolute():
            save_dir = Path(__file__).resolve().parent.parent.parent / save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        output_path = save_dir / output_name

        if not isinstance(self._data, pd.DataFrame):
            raise ValueError("Saving non-DataFrame data is not supported")
        writers[ext](output_path)
        logger.info("Data saved to %s", output_path)


class PandasHandler(DataHandler):
    """
    A class to import historical forecast data from a pandas DataFrame.

    Attributes:
        _data (pd.DataFrame): The DataFrame containing the data.
    """
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.filename: str | None = cfg.datasets.get("filename")
        self.filepath: str | None = cfg.datasets.get("filepath")
        self.arguments: Dict[str, Any] = dict(cfg.datasets.get("load_args", {}))
        self._data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """Load data from the specified file into a pandas DataFrame."""
        data_path = Path(self.filepath) / self.filename
        if not data_path.is_absolute():
            data_path = Path(__file__).resolve().parent.parent.parent / data_path

        logger.info("Loading data from: %s", data_path)

        if not data_path.is_file():
            raise FileNotFoundError(f"File '{self.filename}' not found in '{self.filepath}'")

        ext = Path(self.filename).suffix.lower()
        if ext == ".csv":
            return pd.read_csv(data_path, **self.arguments)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(data_path, **self.arguments)
        elif ext == ".json":
            return pd.read_json(data_path, **self.arguments)
        elif ext == ".parquet":
            engine = self.arguments.pop("engine", None)
            return pd.read_parquet(data_path, engine=engine, **self.arguments)
        else:
            raise ValueError(f"Unsupported file format: {ext}")


class OpenMeteoHandler(DataHandler):
    """

    A class to import historical forecast data from OpenMeteo API.

    Attributes:
        api_url (str): The OpenMeteo API endpoint URL.
        parameters (Dict[str, Any]): Parameters for the API request.

    """
    def __init__(self, api_url,**params: Any):
        # Only import if not already loaded
        super().__init__()
        self.api_url = api_url
        self.parameters: Dict[str, Any] = params
        self._data: pd.DataFrame = self._fetch_data()

    def _fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical forecast data from OpenMeteo API and return as a pandas DataFrame.

        Returns:
            pd.DataFrame: Data fetched from the API.

        Raises:
            Exception: If the API request fails.
            ValueError: If no 'hourly' variables are specified in parameters.
        """
        # Setup the Open-Meteo API client with retry on error
        retry_session = retry(requests.Session(), retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        try:
            print("Fetching data from OpenMeteo API...")
            responses = openmeteo.weather_api(self.api_url,
                                              params=self.parameters)
        except Exception as e:
            print(f"[Error] fetching data from OpenMeteo API: {str(e)}")
            raise
        try:
            variables = self.parameters.get("hourly", [])
            print(f"Requested variables: {variables}")
        except KeyError as e:
            raise ValueError(
                "No 'hourly' variables specified in parameters.") from e
        print(f"Number of responses: {len(responses)}")

        # Empty DataFrame with hourly datetime index
        try:
            start = pd.to_datetime(self.parameters.get("start_date"))
            end = pd.to_datetime(self.parameters.get("end_date"))
        except Exception as e:
            raise ValueError(f"Invalid start_date or end_date in parameters: {e}")

        hourly_index = pd.date_range(start=start, end=end, freq='h', tz='UTC')
        hourly_data = pd.DataFrame({"datetime": hourly_index})

        # Loop through responses and extract data
        for response in responses:
            lat = response.Latitude()
            lon = response.Longitude()
            print(
                f"\nCoordinates: {lat}°N "
                f"{lon}°E"
            )
            print(
                f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s"
            )

            # Process hourly data. The order of variables needs to be the same
            # as requested.
            hourly = response.Hourly()
            response_data = {"date": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            )}
            for i, var_name in enumerate(variables):
                var = hourly.Variables(i).ValuesAsNumpy()
                response_data[var_name] = var
            response_df = pd.DataFrame(data=response_data)
            # build suffix from response coordinates
            suffix = f"_{lat}N_{lon}E"
            # rename all variable columns except the datetime column
            rename_map = {
                col: f"{col}{suffix}" for col in response_df.columns
                if col != 'date'
            }
            response_df = response_df.rename(columns=rename_map)

            # Merge response data into hourly_data
            hourly_data = hourly_data.merge(
                response_df, left_on='datetime', right_on='date', how='left'
            ).drop(columns=['date'])

        return hourly_data


class EntsoeHandler(DataHandler):
    """
    A class to import historical forecast data from ENTSO-E API.

    Attributes:
        client (EntsoePandasClient): The ENTSO-E API client instance.
        parameters (Dict[str, Any]): Parameters for the API request.

    """
    def __init__(self, query: str, country_code: str, include_neighbours: bool = False, **params: Any):
        # Only import if not already loaded
        super().__init__()
        self.client = EntsoePandasClient(api_key="b9ef1885-8f52-4955-be69-dda321d07390")
        self.parameters: Dict[str, Any] = params
        self.query = query
        self.country_code = country_code
        self.include_neighbours = include_neighbours
        self._fetch_data()

    def _fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical forecast data from ENTSO-E API and return as a pandas DataFrame.

        Returns:
            pd.DataFrame: Data fetched from the API.
        """
        neighbours = mappings.NEIGHBOURS.get(self.country_code, [])
        country_tuples = [(self.country_code, v) for v in neighbours] + [(v, self.country_code) for v in neighbours]


        query_name = "query_" + self.query.replace(" ", "_").lower()
        logger.info(f"Fetching {query_name} from ENTSO-E API...")
        method = getattr(self.client, query_name, None)
        if method is None:
            raise AttributeError(f"{query_name} is not valid")


        valid_params = inspect.signature(method).parameters

        kwargs = {
            k: v
            for k, v in self.parameters.items()
            if k in valid_params and v is not None
        }

        if "country_code" in valid_params and "country_code" not in kwargs:
            kwargs["country_code"] = self.country_code

        sig = inspect.signature(method)

        required_params = [
            name
            for name, param in sig.parameters.items()
            if param.default is inspect.Parameter.empty
            and param.kind not in (
                inspect.Parameter.VAR_POSITIONAL,   # *args
                inspect.Parameter.VAR_KEYWORD       # **kwargs
            )
        ]
        missing_params = [p for p in required_params if p not in kwargs]
        if missing_params:
            raise ValueError(
                f"Missing required parameters for {query_name}: {missing_params}"
            )

        if "country_code_from" in valid_params:
            data = []
            if kwargs["country_code_from"] == "all" and kwargs["country_code_to"] == "all":
                tuples = country_tuples
            elif kwargs["country_code_to"] == "all":
                values_country_from = list(kwargs.get("country_code_from"))

                if isinstance(values_country_from, list) and all(isinstance(item, str) for item in values_country_from):
                    # It's a list of strings
                    tuples = [(self.country_code, v) for v in neighbours] + [(v, self.country_code) for v in values_country_from]
                else:
                    raise TypeError("country_code_from must be a list of strings")
            elif kwargs["country_code_from"] == "all":
                values_country_to = list(kwargs.get("country_code_to"))

                if isinstance(values_country_to, list) and all(isinstance(item, str) for item in values_country_to):
                    # It's a list of strings
                    tuples = [(self.country_code, v) for v in values_country_to] + [(v, self.country_code) for v in neighbours]
                else:
                    raise TypeError("country_code_to must be a list of strings")
            else:
                values_country_from = list(kwargs.get("country_code_from"))
                values_country_to = list(kwargs.get("country_code_to"))
                if (isinstance(values_country_from, list) and all(isinstance(item, str) for item in values_country_from) and
                    isinstance(values_country_to, list) and all(isinstance(item, str) for item in values_country_to)):
                    # Both are lists of strings
                    tuples = [(self.country_code, v) for v in values_country_to] + [(v, self.country_code) for v in values_country_from]
                else:
                    raise TypeError("country_code_from and country_code_to must be lists of strings")

            for t in tuples:
                logger.info(f"From country to country: {t[0]} -> {t[1]}")
                kwargs["country_code_from"] = t[0]
                kwargs["country_code_to"] = t[1]
                try:
                    result = method(
                        **kwargs
                    )
                    if isinstance(result, pd.DataFrame):
                        result.columns = [f"{col}_{t[0]}_{t[1]}_" for col in result.columns]
                        data.append(result)
                    elif isinstance(result, pd.Series):
                        result.name = f"{t[0]}_{t[1]}"
                        data.append(result.to_frame())
                    else:
                        logger.error(f"Unexpected result type for {query_name} for tuple: {t}, expected DataFrame or Series, got {type(result)}")
                        continue
                except Exception as e:
                    logger.error(f"Error fetching data {query_name} for: {t[0]} -> {t[1]}, error: {e}")
                    continue
            if not data:
                raise ValueError(f"No data fetched for {query_name} with country_code_from/to combinations.")
            df_total = data[0]
            for df in data[1:]:
                df_total = df_total.merge(df, left_index=True, right_index=True, how='outer')
            self.set_data(df_total)
        elif self.include_neighbours:
            BZNs = [self.country_code] + neighbours
            data = []
            for t in BZNs:
                logger.info(f"Country: {t}")
                kwargs["country_code"] = t
                try:
                    result = method(
                        **kwargs
                    )
                    if isinstance(result, pd.DataFrame):
                        result.columns = [f"{col}_{t}" for col in result.columns]
                        data.append(result)
                    elif isinstance(result, pd.Series):
                        result.name = f"{t}"
                        data.append(result.to_frame())
                    else:
                        logger.error(f"Unexpected result type for {query_name} for country: {t}, expected DataFrame or Series, got {type(result)}")
                        continue
                except Exception as e:
                    logger.error(f"Error fetching data {query_name} for country: {t}, error: {e}")
                    continue
            if not data:
                raise ValueError(f"No data fetched for {query_name} for country '{self.country_code}' and its neighbours.")
            df_total = data[0]
            for df in data[1:]:
                df_total = df_total.merge(df, left_index=True, right_index=True, how='outer')
            self.set_data(df_total)
        else:
            try:
                self.set_data(pd.DataFrame(method(
                    **kwargs
                )))
            except Exception as e:
                logger.error(f"Error fetching data {query_name} for country: {self.country_code}, error: {e}")
                raise


class EnerginetHandler(DataHandler):
    """
    A class to import data from Energinet API.
    """
    BASE_URL = "https://api.energidataservice.dk/dataset"
    def __init__(self, dataset_name: str, start: pd.Timestamp | datetime | str = None, end: pd.Timestamp | datetime | str = None, columns: list[str] = None, filter: dict = None, sort: list[str] = None, limit: int = None):
        # Only import if not already loaded
        super().__init__()
        self.dataset_name = dataset_name
        self.start = start
        self.end = end
        self.columns = columns
        self.filter = filter
        self.sort = sort
        self.limit = limit
        self.api_url = f"{self.BASE_URL}/{self.dataset_name}"
        self._data: pd.DataFrame = self._fetch_data()

    def _fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical forecast data from Energinet API and return as a pandas DataFrame.

        Returns:
            pd.DataFrame: Data fetched from the API.

        Raises:
            Exception: If the API request fails.
            ValueError: If no 'hourly' variables are specified in parameters.
        """

        params: Dict[str, Any] = {}

        # Time range
        if self.start is not None:
            params["start"] = _to_iso(self.start)

        if self.end is not None:
            params["end"] = _to_iso(self.end)

        # Column selection
        if self.columns:
            params["columns"] = ",".join(self.columns)

        # Sorting
        if self.sort:
            params["sort"] = ",".join(self.sort)

        # Row limit
        if self.limit:
            params["limit"] = self.limit

        # Filters (Energinet expects JSON formatted filter)
        if self.filter:
            filter_dict = (
                OmegaConf.to_container(self.filter, resolve=True)
                if not isinstance(self.filter, dict)
                else self.filter
            )
            params["filter"] = json.dumps(filter_dict)

        response = requests.get(self.api_url, params=params)

        if response.status_code != 200:
            raise Exception(
                f"API request failed with status {response.status_code}: {response.text}"
            )

        json_data = response.json()

        # Energidataservice format: data is inside "records"
        records = json_data.get("records", [])

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)

        return df

class NetztransparentHandler(DataHandler):
    """
    A class to import data from the Netztransparenz.de WebAPI.

    The Netztransparenz WebAPI is jointly operated by the four German TSOs
    (50Hertz, Amprion, TenneT, TransnetBW). Access uses an OAuth2
    client-credentials flow: a short-lived bearer token is requested from the
    identity service and then used to query the data service. Data endpoints
    return semicolon-separated CSV using German number formatting (comma as the
    decimal separator), which is parsed into a pandas DataFrame.

    Credentials (ClientID/ClientSecret) are created in the WebAPI portal and are
    read from the ``IPNT_CLIENT_ID`` / ``IPNT_CLIENT_SECRET`` environment
    variables by default, or can be passed directly.

    Swagger:
        https://extranet.netztransparenz.de/DesktopModules/LotesDataManagementExtranet/Swagger/index.html?version=public

    Attributes:
        endpoint (str): Data path after ``/api/v1/data/`` (e.g. "prognose/solar").
        start (pd.Timestamp | datetime | str | None): Start of the time range.
        end (pd.Timestamp | datetime | str | None): End of the time range.
        read_args (Dict[str, Any]): Extra keyword arguments for ``pd.read_csv``.
    """

    TOKEN_URL = "https://identity.netztransparenz.de/users/connect/token"
    BASE_URL = "https://ds.netztransparenz.de/api/v1"

    def __init__(
        self,
        endpoint: str,
        start: pd.Timestamp | datetime | str | None = None,
        end: pd.Timestamp | datetime | str | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        read_args: dict | None = None,
    ) -> None:
        super().__init__()
        self.endpoint = endpoint.strip("/")
        self.start = start
        self.end = end
        self.client_id = client_id or os.getenv("IPNT_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("IPNT_CLIENT_SECRET")
        self.read_args: Dict[str, Any] = dict(read_args or {})
        self._data: pd.DataFrame = self._fetch_data()

    @staticmethod
    def _format_date(value: pd.Timestamp | datetime | str) -> str:
        """
        Format a date as the API expects: ``YYYY-MM-DDTHH:MM:SS`` (no timezone).

        Args:
            value: Date to format (Timestamp, datetime or parseable string).

        Returns:
            str: ISO-like datetime string without timezone offset.
        """
        return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%S")

    def _get_access_token(self) -> str:
        """
        Obtain an OAuth2 access token via the client-credentials grant.

        Returns:
            str: A bearer access token (valid ~1 hour).

        Raises:
            ValueError: If credentials are missing.
            Exception: If the token request fails or returns no token.
        """
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Netztransparenz API credentials missing. Set IPNT_CLIENT_ID and "
                "IPNT_CLIENT_SECRET environment variables, or pass client_id and "
                "client_secret."
            )

        logger.info("Requesting Netztransparenz access token...")
        response = requests.post(
            self.TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=30,
        )

        if response.status_code != 200:
            raise Exception(
                f"Failed to obtain Netztransparenz access token "
                f"(status {response.status_code}): {response.text}"
            )

        token = response.json().get("access_token")
        if not token:
            raise Exception(
                "No access_token returned by the Netztransparenz identity service."
            )
        return token

    def _fetch_data(self) -> pd.DataFrame:
        """
        Fetch data from the Netztransparenz API and return it as a DataFrame.

        Builds the request URL as
        ``/api/v1/data/{endpoint}/{dateFrom}/{dateTo}`` (the date range is
        appended only when both ``start`` and ``end`` are provided), sends the
        bearer token and parses the CSV response.

        Returns:
            pd.DataFrame: Data fetched from the API (empty if no content).

        Raises:
            Exception: If the API request fails.
        """
        token = self._get_access_token()

        url = f"{self.BASE_URL}/data/{self.endpoint}"
        if self.start is not None and self.end is not None:
            date_from = quote(self._format_date(self.start), safe="")
            date_to = quote(self._format_date(self.end), safe="")
            url = f"{url}/{date_from}/{date_to}"

        logger.info("Fetching data from Netztransparenz API: %s", url)
        response = requests.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=120,
        )

        if response.status_code != 200:
            raise Exception(
                f"Netztransparenz API request failed with status "
                f"{response.status_code}: {response.text}"
            )

        # A handful of endpoints reply with JSON; data endpoints return CSV.
        content_type = response.headers.get("Content-Type", "")
        if "json" in content_type.lower():
            return pd.json_normalize(response.json())

        text = response.text
        if not text.strip():
            logger.warning("Netztransparenz API returned an empty response for %s", url)
            return pd.DataFrame()

        # German CSV: ';' separator and ',' as decimal separator.
        read_args = {"sep": ";", "decimal": ",", **self.read_args}
        return pd.read_csv(StringIO(text), **read_args)

def _to_iso(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)

def _ensure_save_dir(save_dir: str | None) -> Path:
    """Resolve ``save_dir`` and create it if it does not exist.

    Relative paths are anchored to the project root, matching ``DataHandler.save``.
    """
    save_dir = Path(save_dir) if save_dir else Path.cwd()
    if not save_dir.is_absolute():
        save_dir = Path(__file__).resolve().parent.parent.parent / save_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config_dev.yaml")
def pull_openmeteo(cfg: DictConfig) -> None:
    params = cfg.datasets.get("parameters")
    API_URL = cfg.datasets.get("api_url")
    weather_data = OpenMeteoHandler(API_URL, **params)
    save_dir = cfg.datasets.get("savepath")
    _ensure_save_dir(save_dir)
    savename = "openmeteo.csv"
    weather_data.save(savename=savename, save_dir=save_dir, index=False)

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config_dev.yaml")
def pull_entsoe(cfg: DictConfig) -> None:
    mapping_process_types = {
        "A47": "_mFRR",
        "A51": "_aFRR",
        "A52": "_FCR"
    }
    BZN =  cfg.datasets.get("BZN")
    queries = list(cfg.datasets.get("queries").keys())
    save_dir = cfg.datasets.get("savepath")
    _ensure_save_dir(save_dir)
    include_neighbours = cfg.datasets.get("include_neighbours", False)
    kwargs = dict(
        start=pd.Timestamp(cfg.datasets.get("start_date"), tz="UTC"),
        end=pd.Timestamp(cfg.datasets.get("end_date"), tz="UTC")
        )


    for q in queries:
        logger.info(f"Starting data import for query '{q}' and zone '{BZN}'...")
        logger.info(f"Query args: {cfg.datasets.queries[q]}")
        query_kwargs = cfg.datasets.queries.get(q) or {}
        process_type = query_kwargs.get("process_type", None)
        if process_type:
            type_ = mapping_process_types.get(process_type, "")
        else:
            type_ = ""
        kwargs_ = kwargs.copy()
        kwargs_.update(query_kwargs)
        try:
            entsoe_handler = EntsoeHandler(
                query=q,
                country_code = BZN,
                include_neighbours=include_neighbours,
                **kwargs_,
            )
        except Exception as e:
            logger.error(f"Failed to fetch data for query '{q}' and zone '{BZN}': {e}")
            continue
        savename = f"{q.replace(' ', '_').lower()}_{BZN}{type_}.csv"
        entsoe_handler.save(savename=savename, save_dir=save_dir, index=True)

@hydra.main(version_base="1.3", config_path="../../configs", config_name="config_dev.yaml")
def pull_energinet(cfg: DictConfig) -> None:
    requests = cfg.datasets.get("requests")
    save_dir = cfg.datasets.get("savepath")
    _ensure_save_dir(save_dir)

    if requests:
        for dataset_name in requests:
            kwargs = requests.get(dataset_name)
            logger.info(f"Starting data import for dataset '{dataset_name}' with args: {kwargs}")
            try:
                energinet_handler = EnerginetHandler(
                    dataset_name=dataset_name,
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Failed to fetch data for dataset '{dataset_name}': {e}")
                continue
            savename = f"{dataset_name}.csv"
            energinet_handler.save(savename=savename, save_dir=save_dir, index=False)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config_dev.yaml")
def pull_netztransparenz(cfg: DictConfig) -> None:
    requests_cfg = cfg.datasets.get("requests")
    save_dir = cfg.datasets.get("savepath")

    if not requests_cfg:
        logger.warning("No 'requests' specified in the Netztransparenz dataset config.")
        return

    for name, kwargs in requests_cfg.items():
        kwargs = dict(kwargs or {})
        endpoint = kwargs.pop("endpoint", None)
        if endpoint is None:
            logger.error(f"Skipping '{name}': no 'endpoint' specified.")
            continue
        logger.info(f"Starting data import for '{name}' (endpoint='{endpoint}') with args: {kwargs}")
        try:
            netztransparenz_handler = NetztransparentHandler(
                endpoint=endpoint,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Failed to fetch data for '{name}': {e}")
            continue
        savename = f"{name}.csv"
        netztransparenz_handler.save(savename=savename, save_dir=save_dir, index=False)


if __name__ == "__main__":
    pull_openmeteo()
