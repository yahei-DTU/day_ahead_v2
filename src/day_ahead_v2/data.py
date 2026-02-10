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

import sys
from pathlib import Path
import logging
from omegaconf import DictConfig
import hydra
from typing import Dict, Any
import numpy as np
import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
from day_ahead_v2.utils.data_validation import (
    ValidationReport, ShapeInfo, MissingValuesInfo, MissingColumnInfo,
    DuplicateInfo, MemoryUsageInfo, NumericSummaryStats
)

# Initialize logger
logger = logging.getLogger(__name__)

class DataHandler:
    """
    Load data from common file formats into a pandas DataFrame.

    Supports: CSV, Excel, JSON, Parquet.

    Can be instantiated empty (no file loaded), or with a filename to auto-load.

    Attributes:
        filename (str | None): The file name loaded or to be loaded.
        filepath (str | None): Directory path containing the file.
        arguments (dict[str, Any]): Extra kwargs forwarded to pandas read_* functions.
        _data (pd.DataFrame): Loaded data (DataFrame).
    """

    def __init__(self, cfg: DictConfig | None = None) -> None:
        """
        Initialize a DataHandler.

        Args:
            cfg (DictConfig | None): Optional configuration for loading data.
        """
        if cfg is not None:
            self.filename: str | None = cfg.datasets.get("filename", None)
            self.filepath: str | None = cfg.datasets.get("filepath", None)
            self.arguments: Dict[str, Any] = dict(cfg.datasets.get("load_args", {}))
            self._data = self._load_data()
        else:
            self.filename: str | None = None
            self.filepath: str | None = None
            self.arguments: Dict[str, Any] = {}
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

    def load_file(self, filename: str, filepath: str , **kwargs) -> None:
        """
        Load data from a file after instance creation.

        Args:
            filename (str): The name of the file to load.
            filepath (str): The path to the directory containing the file.
            **kwargs: Additional arguments to pass to the pandas read function.
        """
        self.filename: str = filename
        self.filepath: str = filepath
        self.arguments.update(kwargs)
        self._data: pd.DataFrame = self._load_data()

    def set_data(self, data: pd.DataFrame, filename: str | None = None) -> None:
        """
        Set data directly without loading from file.

        Args:
            data (pd.DataFrame): The data to set.
            filename (str | None): Optional filename for reference.
        """
        self._data: pd.DataFrame = data
        if filename is not None:
            self.filename = filename

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

        self._label_mapping(df, cfg.datasets, log)
        self._drop_columns(df, cfg.datasets, log)
        self._rename_columns(df, cfg.datasets, log)
        self._drop_duplicates(df, cfg.datasets, log)
        self._drop_missing_columns(df, cfg.datasets, log)
        self._fill_missing(df, cfg.datasets, log)
        self._convert_types(df, cfg.datasets, log)
        self._normalize_columns(df, cfg.datasets, log)
        self._to_dtype_float32(df)

        return self._build_new_handler(df, log)

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

            # Convert mapping keys & values to int (robust to string/numeric columns)
            try:
                mapping_int = {int(k): int(v) for k, v in mapping_dict.items()}
            except ValueError as e:
                raise ValueError(
                    f"Invalid label mapping for column '{column}'. "
                    f"Keys and values must be castable to int."
                ) from e

            # Convert column to numeric (safe even if already numeric)
            series = pd.to_numeric(df[column], errors="coerce")

            # Check for unseen labels
            unique_labels = set(series.dropna().unique())
            expected_labels = set(mapping_int.keys())
            unseen = unique_labels - expected_labels

            if unseen:
                raise ValueError(
                    f"Unmapped labels found in column '{column}': {sorted(unseen)}. "
                    f"Expected only: {sorted(expected_labels)}"
                )

            # Apply mapping
            df[column] = series.map(mapping_int).astype("int64")

            msg = (
                f"Applied label mapping to '{column}': "
                f"{mapping_int}"
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
                if hasattr(cfg.training, "feature_columns") and cfg.training.feature_columns:
                    original_features = list(cfg.training.feature_columns)
                    cfg.training.feature_columns = [
                        col for col in cfg.training.feature_columns if col not in cols_to_drop
                    ]
                    removed = set(original_features) - set(cfg.training.feature_columns)
                    if removed:
                        log.append(f"Removed from feature_columns in config: {list(removed)}")
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

        elif method == "backward":
            df.bfill(limit=limit, inplace=True)

        elif method == "zero":
            df.fillna(0, inplace=True)

        elif method == "interpolate":
            if limit is None:
                df[numeric_cols] = df[numeric_cols].interpolate()
            else:
                df[numeric_cols] = df[numeric_cols].interpolate(limit=limit)
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
            raise ValueError(f"Column '{datetime_column}' is not of datetime type.")

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

    def save(self, cfg, **kwargs) -> None:
        """
        Save the data to disk. Supports CSV, Parquet, Excel, JSON formats.

        Args:
            filename (str | None): Name of the output file. If None, defaults to
                original filename with '_saved' suffix and '.csv' extension.
            directory (str | None): Directory to save the file. If None, uses
                the original file's directory or current working directory.
            **kwargs: Additional arguments passed to the pandas to_* functions.

        Raises:
            ValueError: If no data is loaded.
            ValueError: If filename does not include an extension or has unsupported format.
            ValueError: If saving non-DataFrame data is attempted.
        Returns:
            Path: The path to the saved file.
        """
        if self._data is None:
            raise ValueError("No data to save. Load data first.")

        # Supported extensions
        writers = {
            ".csv": lambda p: self._data.to_csv(p, index=False, **kwargs),
            ".parquet": lambda p: self._data.to_parquet(p, index=False, **kwargs),
            ".xlsx": lambda p: self._data.to_excel(p, index=False, **kwargs),
            ".xls": lambda p: self._data.to_excel(p, index=False, **kwargs),
            ".json": lambda p: self._data.to_json(p, orient="records", indent=2, **kwargs),
        }
        savename = cfg.datasets.get("savename", None)
        save_dir = cfg.datasets.get("savepath", None)

        # Resolve filename
        if savename:
            output_name = Path(savename)
        else:
            base = Path(self.filename).stem if self.filename else "data"
            output_name = Path(f"{base}_saved.csv")

        ext = output_name.suffix.lower()
        if not ext:
            raise ValueError("Filename must include a file extension")
        if ext not in writers:
            raise ValueError(f"Unsupported file format: {ext}")

        # Resolve directory
        if save_dir:
            save_dir = Path(save_dir)
        elif self.filepath:
            save_dir = Path(self.filepath).parent.parent
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
        return


class OpenMeteoHandler(DataHandler):
    """

    A class to import historical forecast data from OpenMeteo API.

    Attributes:
        api_url (str): The OpenMeteo API endpoint URL.
        parameters (Dict[str, Any]): Parameters for the API request.

    """
    def __init__(self, api_url,**params: Any):
        # Only import if not already loaded
        super().__init__(filename=None)
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
        # Setup the Open-Meteo API client with cache and retry on error
        cache_session = requests_cache.CachedSession('.cache',
                                                     expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
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


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config_dev.yaml")
def main(cfg: DictConfig) -> None:
    elspotprices = pd.read_csv("data/raw/Elspotprices.csv", sep=';', decimal=',')
    elspotprices = elspotprices.drop(columns=['HourDK', 'PriceArea','SpotPriceDKK'], errors='ignore')
    elspotprices = elspotprices.rename(columns={'SpotPriceEUR': 'SpotPriceEUR_DK1'})
    print(elspotprices.head())
    elspotprices["HourUTC"] = pd.to_datetime(elspotprices["HourUTC"], format='%Y-%m-%d %H:%M:%S', utc=True)
    wind_data = pd.read_csv("data/raw/Enfor_DA_wind_power_forecast.csv")
    wind_data = pd.read_csv("data/raw/Enfor_DA_wind_power_forecast.csv")
    wind_data["Time_begin"] = pd.to_datetime(wind_data["Time_begin"], format='%Y-%m-%d %H:%M:%S', utc=True)
    # check for duplicate Time_begin values
    duplicates = wind_data["Time_begin"].duplicated().sum()
    # print both rows with duplicate Time_begin values
    print(f"Number of duplicate Time_begin values: {duplicates}")
    # Keep only the first row for each duplicate Time_begin value
    wind_data = wind_data.drop_duplicates(subset='Time_begin', keep='first')
    wind_data = wind_data.drop(columns=['Time_end', 'SCADAPowerMeas','PTime'], errors='ignore')
    wind_data = wind_data.rename(columns={'Time_begin': 'Hour', 'PowerPred': 'WindFarm_WindPowerForecast', 'SettlementPowerMeas': 'WindFarm_ActualWindPower'})
    for col in wind_data.columns:
        if col not in ["Hour", "PTime"]:
            wind_data[col] = pd.to_numeric(wind_data[col],
                                                errors='coerce')
    param_data = elspotprices.merge(
            wind_data, left_on='HourUTC',
            right_on='Hour', how='left').drop(columns=['Hour'])

    imbalance_data = pd.read_parquet("data/processed/imbalance_data.parquet", engine='pyarrow')
    # keep only column ImbalancePriceEUR_DK1
    imbalance_data["datetime"] = pd.to_datetime(
        imbalance_data["datetime"],
        format='%Y-%m-%d %H:%M:%S%z'
    )
    full_data = imbalance_data.merge(param_data, left_on='datetime', right_on='HourUTC', how='left').drop(columns=['HourUTC'])
    full_data.to_csv("data/processed/optimization_parameter.csv", index=False)
    full_data.to_parquet("data/processed/optimization_parameter.parquet", index=False, engine='pyarrow')


if __name__ == "__main__":
    main()

    sys.exit()
    # Create an empty DataFrame with hourly datetime index
    features_df = pd.DataFrame()
    features_df["datetime"] = pd.date_range(start='2023-01-01',
                                            end='2025-09-22',
                                            freq='h', tz='UTC')

    ###########################################################################

    print("=" * 60)
    print("Importing Data: Actual Production DK1")
    print("=" * 60)

    # Import data and validate
    actual_production_DK1 = DataHandler()

    actual_production_2023_DK1 = DataHandler(
        'Actual Generation per Production Type_2023 - DK1.csv')
    actual_production_2024_DK1 = DataHandler(
        'Actual Generation per Production Type_2024 - DK1.csv')
    actual_production_2025_DK1 = DataHandler(
        'Actual Generation per Production Type_2025 - DK1.csv')

    actual_production_DK1.set_data(
        pd.concat([
            actual_production_2023_DK1.data,
            actual_production_2024_DK1.data,
            actual_production_2025_DK1.data
        ], ignore_index=True)
    )

    del (actual_production_2023_DK1,
         actual_production_2024_DK1,
         actual_production_2025_DK1)

    # Transform data: Drop 'Area' column, handle missing values,
    actual_production_DK1 = actual_production_DK1.transform_data(
        drop_columns=['Area'])
    for col in actual_production_DK1.data.columns:
        if col != "MTU":
            actual_production_DK1.data[col] = pd.to_numeric(
                actual_production_DK1.data[col], errors='coerce')
            actual_production_DK1.data.rename(columns={col: f"{col}_DK1"},
                                              inplace=True)
    actual_production_DK1.data["MTU"] = (
        actual_production_DK1.data["MTU"]
        .replace(" (UTC)", "", regex=False).str.split(" - ").str[0].
        pipe(pd.to_datetime, format='%d.%m.%Y %H:%M', utc=True)
    )
    actual_production_DK1.set_data(
        actual_production_DK1.data[
            actual_production_DK1.data["MTU"] <= pd.Timestamp("2025-09-22",
                                                              tz="UTC")
        ]
    )

    # Check for duplicate MTU values
    datetime_duplicates = actual_production_DK1.data["MTU"].duplicated().sum()
    print(f"Number of duplicate MTU values: {datetime_duplicates}")

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        actual_production_DK1 = actual_production_DK1.transform_data(
            drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        actual_production_DK1.set_data(actual_production_DK1.data
                                       .set_index("MTU").resample("h")
                                       .sum(min_count=1))

        # Create lagged features (48 hours)
        actual_production_DK1.set_data(actual_production_DK1.data.shift(48))
        actual_production_DK1.data.columns = [f"{col}_Lag48h" for col in actual_production_DK1.data.columns]

        # Reset index to a column
        actual_production_DK1.set_data(actual_production_DK1.
                                       data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            actual_production_DK1.data, left_on='datetime',
            right_on='MTU', how='left').drop(columns=['MTU'])
    else:
        print("[WARNING] Duplicate MTU values found. "
              "Please check the data for inconsistencies.")

    actual_production_DK1.info()
    del actual_production_DK1

    ###########################################################################

    print("=" * 60)
    print("Importing Data: Actual Production DK2")
    print("=" * 60)

    # Import data and validate
    actual_production_DK2 = DataHandler()

    actual_production_2023_DK2 = DataHandler(
        'Actual Generation per Production Type_2023 - DK2.csv')
    actual_production_2024_DK2 = DataHandler(
        'Actual Generation per Production Type_2024 - DK2.csv')
    actual_production_2025_DK2 = DataHandler(
        'Actual Generation per Production Type_2025 - DK2.csv')

    actual_production_DK2.set_data(
        pd.concat([
            actual_production_2023_DK2.data,
            actual_production_2024_DK2.data,
            actual_production_2025_DK2.data
        ], ignore_index=True)
    )

    del (actual_production_2023_DK2,
         actual_production_2024_DK2,
         actual_production_2025_DK2)

    # Transform data: Drop 'Area' column, handle missing values
    actual_production_DK2 = actual_production_DK2.transform_data(
        drop_columns=['Area'])
    for col in actual_production_DK2.data.columns:
        if col != "MTU":
            actual_production_DK2.data[col] = pd.to_numeric(
                actual_production_DK2.data[col], errors='coerce')
            actual_production_DK2.data.rename(columns={col: f"{col}_DK2"},
                                              inplace=True)
    actual_production_DK2.data["MTU"] = (
        actual_production_DK2.data["MTU"]
        .replace(" (UTC)", "", regex=False).str.split(" - ").str[0].
        pipe(pd.to_datetime, format='%d.%m.%Y %H:%M', utc=True)
    )
    actual_production_DK2.set_data(
        actual_production_DK2.data[
            actual_production_DK2.data["MTU"] <= pd.Timestamp("2025-09-22", tz="UTC")
        ]
    )
    # Check for duplicate MTU values
    datetime_duplicates = actual_production_DK2.data["MTU"].duplicated().sum()
    print(f"Number of duplicate MTU values: {datetime_duplicates}")

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        actual_production_DK2 = actual_production_DK2.transform_data(
            drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        actual_production_DK2.set_data(actual_production_DK2.data
                                       .set_index("MTU").resample("h")
                                       .sum(min_count=1))

        # Create lagged features (48 hours)
        actual_production_DK2.set_data(actual_production_DK2.data.shift(48))
        actual_production_DK2.data.columns = [f"{col}_Lag48h" for col in actual_production_DK2.data.columns]


        # Reset index to a column
        actual_production_DK2.set_data(actual_production_DK2.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            actual_production_DK2.data, left_on='datetime',
            right_on='MTU', how='left').drop(columns=['MTU'])
    else:
        print("[WARNING] Duplicate MTU values found. "
              "Please check the data for inconsistencies.")

    actual_production_DK2.info()
    del actual_production_DK2
    print("Shape features: ", features_df.shape)

    ###########################################################################

    print("=" * 60)
    print("Importing Data: AfrrReservesNordic")
    print("=" * 60)

    # Import data and validate
    afrr_reserves = DataHandler("AfrrReservesNordic.csv", sep=';', decimal=',')
    afrr_reserves_DK1 = DataHandler()
    afrr_reserves_DK2 = DataHandler()

    # Transform data: Drop 'HourDK' column, handle missing values
    afrr_reserves = afrr_reserves.transform_data(
        drop_columns=['HourDK',
                      'aFRR_DownCapPriceDKK',
                      'aFRR_UpCapPriceDKK'])
    for col in afrr_reserves.data.columns:
        for col in afrr_reserves.data.columns:
            if col not in ["HourUTC", "PriceArea"]:
                afrr_reserves.data[col] = pd.to_numeric(afrr_reserves.data[col],
                                                        errors='coerce')
    afrr_reserves.data["HourUTC"] = (
        afrr_reserves.data["HourUTC"]
        .pipe(pd.to_datetime, format='%Y-%m-%d %H:%M:%S', utc=True)
    )
    afrr_reserves.set_data(
        afrr_reserves.data[
            afrr_reserves.data["HourUTC"] <= pd.Timestamp("2025-09-22", tz="UTC")
        ]
    )

    # Split data into DK1 and DK2
    afrr_reserves_DK1.set_data(
        afrr_reserves.data[afrr_reserves.data["PriceArea"] == "DK1"]
        .drop(columns=["PriceArea"])
    )

    afrr_reserves_DK2.set_data(
        afrr_reserves.data[afrr_reserves.data["PriceArea"] == "DK2"]
        .drop(columns=["PriceArea"])
    )

    del afrr_reserves

    # DK1 data processing
    # Check for duplicate HourUTC values
    datetime_duplicates = afrr_reserves_DK1.data["HourUTC"].duplicated().sum()
    if datetime_duplicates > 0:
        raise ValueError(
            f"Duplicate HourUTC values found: {datetime_duplicates}")

    # Change column names to indicate DK1
    for col in afrr_reserves_DK1.data.columns:
        if col != "HourUTC":
            afrr_reserves_DK1.data.rename(columns={col: f"{col}_DK1"},
                                          inplace=True)

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        afrr_reserves_DK1 = afrr_reserves_DK1.transform_data(
            drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        afrr_reserves_DK1.set_data(afrr_reserves_DK1.data
                                   .set_index("HourUTC").resample("h")
                                   .sum(min_count=1))
        # Reset index to a column
        afrr_reserves_DK1.set_data(afrr_reserves_DK1.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            afrr_reserves_DK1.data, left_on='datetime',
            right_on='HourUTC', how='left').drop(columns=['HourUTC'])
    else:
        print("[WARNING] Duplicate HourUTC values found. "
              "Please check the data for inconsistencies.")

    del afrr_reserves_DK1

    # DK2 data processing
    # Check for duplicate HourUTC values
    datetime_duplicates = afrr_reserves_DK2.data["HourUTC"].duplicated().sum()
    if datetime_duplicates > 0:
        raise ValueError(f"Duplicate HourUTC values found: {datetime_duplicates}")

    # Change column names to indicate DK2
    for col in afrr_reserves_DK2.data.columns:
        if col != "HourUTC":
            afrr_reserves_DK2.data.rename(columns={col: f"{col}_DK2"}, inplace=True)

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        afrr_reserves_DK2 = afrr_reserves_DK2.transform_data(
            drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        afrr_reserves_DK2.set_data(afrr_reserves_DK2.data
                                   .set_index("HourUTC").resample("h")
                                   .sum(min_count=1))
        # Reset index to a column
        afrr_reserves_DK2.set_data(afrr_reserves_DK2.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            afrr_reserves_DK2.data, left_on='datetime',
            right_on='HourUTC', how='left').drop(columns=['HourUTC'])
    else:
        print("[WARNING] Duplicate HourUTC values found. "
              "Please check the data for inconsistencies.")

    afrr_reserves_DK2.info()
    del afrr_reserves_DK2
    print("Shape features: ", features_df.shape)

    ###########################################################################

    print("=" * 60)
    print("Importing Data: Forecasted Transfer Capacities - DK1 to DE-LU")
    print("=" * 60)

    # Import data and validate
    forecasted_transfer_capacities_DK1_DE_LU = DataHandler()

    forecasted_transfer_capacities_DK1_DE_LU_2023 = DataHandler(
        'Forecasted Transfer Capacities - Day Ahead_2023 - DK1 to DE-LU.csv')
    forecasted_transfer_capacities_DK1_DE_LU_2024 = DataHandler(
        'Forecasted Transfer Capacities - Day Ahead_2024 - DK1 to DE-LU.csv')
    forecasted_transfer_capacities_DK1_DE_LU_2025 = DataHandler(
        'Forecasted Transfer Capacities - Day Ahead_2025 - DK1 to DE-LU.csv')

    forecasted_transfer_capacities_DK1_DE_LU.set_data(
        pd.concat([
            forecasted_transfer_capacities_DK1_DE_LU_2023.data,
            forecasted_transfer_capacities_DK1_DE_LU_2024.data,
            forecasted_transfer_capacities_DK1_DE_LU_2025.data
        ], ignore_index=True)
    )

    del (forecasted_transfer_capacities_DK1_DE_LU_2023,
         forecasted_transfer_capacities_DK1_DE_LU_2024,
         forecasted_transfer_capacities_DK1_DE_LU_2025)

    # Transform data: Drop 'Area' column, handle missing values
    forecasted_transfer_capacities_DK1_DE_LU = (
        forecasted_transfer_capacities_DK1_DE_LU.transform_data(
            drop_columns=['Area'])
    )
    for col in forecasted_transfer_capacities_DK1_DE_LU.data.columns:
        if col != "Time (UTC)":
            forecasted_transfer_capacities_DK1_DE_LU.data[col] = (
                pd.to_numeric(forecasted_transfer_capacities_DK1_DE_LU
                              .data[col], errors='coerce')
            )
    forecasted_transfer_capacities_DK1_DE_LU.data["Time (UTC)"] = (
        forecasted_transfer_capacities_DK1_DE_LU.data["Time (UTC)"]
        .str.split(" - ").str[0].
        pipe(pd.to_datetime, format='%d.%m.%Y %H:%M', utc=True)
    )
    forecasted_transfer_capacities_DK1_DE_LU.set_data(
        forecasted_transfer_capacities_DK1_DE_LU.data[
            forecasted_transfer_capacities_DK1_DE_LU.data["Time (UTC)"] <= (
                pd.Timestamp("2025-09-22", tz="UTC"))
        ]
    )
    # Check for duplicate Time (UTC) values
    datetime_duplicates = (forecasted_transfer_capacities_DK1_DE_LU.
                           data["Time (UTC)"].duplicated().sum())
    print(f"Number of duplicate Time (UTC) values: {datetime_duplicates}")

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        forecasted_transfer_capacities_DK1_DE_LU = (
            forecasted_transfer_capacities_DK1_DE_LU
            .transform_data(drop_missing_threshold=0.1)
        )

        # Resample to hourly frequency, summing values within each hour
        forecasted_transfer_capacities_DK1_DE_LU.set_data(
            forecasted_transfer_capacities_DK1_DE_LU
            .data.set_index("Time (UTC)").resample("h").sum(min_count=1))
        # Reset index to a column
        forecasted_transfer_capacities_DK1_DE_LU.set_data(
            forecasted_transfer_capacities_DK1_DE_LU.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            forecasted_transfer_capacities_DK1_DE_LU.data, left_on='datetime',
            right_on='Time (UTC)', how='left').drop(columns=['Time (UTC)'])
    else:
        print("[WARNING] Duplicate Time (UTC) values found. "
              "Please check the data for inconsistencies.")

    forecasted_transfer_capacities_DK1_DE_LU.info()

    del forecasted_transfer_capacities_DK1_DE_LU
    print("Shape features: ", features_df.shape)

    ###########################################################################

    print("=" * 60)
    print("Importing Data: Forecasted Transfer Capacities - DK1 to NL")
    print("=" * 60)

    # Import data and validate
    forecasted_transfer_capacities_DK1_NL = DataHandler()

    forecasted_transfer_capacities_DK1_NL_2023 = DataHandler(
        'Forecasted Transfer Capacities - Day Ahead_2023 - DK1 to NL.csv')
    forecasted_transfer_capacities_DK1_NL_2024 = DataHandler(
        'Forecasted Transfer Capacities - Day Ahead_2024 - DK1 to NL.csv')
    forecasted_transfer_capacities_DK1_NL_2025 = DataHandler(
        'Forecasted Transfer Capacities - Day Ahead_2025 - DK1 to NL.csv')

    forecasted_transfer_capacities_DK1_NL.set_data(
        pd.concat([
            forecasted_transfer_capacities_DK1_NL_2023.data,
            forecasted_transfer_capacities_DK1_NL_2024.data,
            forecasted_transfer_capacities_DK1_NL_2025.data
        ], ignore_index=True)
    )

    del (forecasted_transfer_capacities_DK1_NL_2023,
         forecasted_transfer_capacities_DK1_NL_2024,
         forecasted_transfer_capacities_DK1_NL_2025)

    # Transform data: Drop 'Area' column, handle missing values
    forecasted_transfer_capacities_DK1_NL = (
        forecasted_transfer_capacities_DK1_NL.transform_data(
            drop_columns=['Area']
        )
    )
    for col in forecasted_transfer_capacities_DK1_NL.data.columns:
        if col != "Time (UTC)":
            forecasted_transfer_capacities_DK1_NL.data[col] = pd.to_numeric(
                forecasted_transfer_capacities_DK1_NL.data[col],
                errors='coerce'
            )
    forecasted_transfer_capacities_DK1_NL.data["Time (UTC)"] = (
        forecasted_transfer_capacities_DK1_NL.data["Time (UTC)"]
        .str.split(" - ").str[0].
        pipe(pd.to_datetime, format='%d.%m.%Y %H:%M', utc=True)
    )
    forecasted_transfer_capacities_DK1_NL.set_data(
        forecasted_transfer_capacities_DK1_NL.data[
            forecasted_transfer_capacities_DK1_NL.data["Time (UTC)"] <= (
                pd.Timestamp("2025-09-22", tz="UTC"))
        ]
    )
    # Check for duplicate Time (UTC) values
    datetime_duplicates = (forecasted_transfer_capacities_DK1_NL
                           .data["Time (UTC)"].duplicated().sum())
    print(f"Number of duplicate Time (UTC) values: {datetime_duplicates}")

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        forecasted_transfer_capacities_DK1_NL = (
            forecasted_transfer_capacities_DK1_NL.transform_data(
                drop_missing_threshold=0.1
            )
        )

        # Resample to hourly frequency, summing values within each hour
        forecasted_transfer_capacities_DK1_NL.set_data(forecasted_transfer_capacities_DK1_NL.data
                                       .set_index("Time (UTC)").resample("h")
                                       .sum(min_count=1))
        # Reset index to a column
        forecasted_transfer_capacities_DK1_NL.set_data(forecasted_transfer_capacities_DK1_NL.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            forecasted_transfer_capacities_DK1_NL.data, left_on='datetime',
            right_on='Time (UTC)', how='left').drop(columns=['Time (UTC)'])
    else:
        print("[WARNING] Duplicate Time (UTC) values found. "
              "Please check the data for inconsistencies.")

    print(forecasted_transfer_capacities_DK1_NL.info())
    del forecasted_transfer_capacities_DK1_NL
    print("Shape features: ", features_df.shape)

    ###########################################################################

    print("=" * 60)
    print("Importing Data: Forecasts_Hour")
    print("=" * 60)

    # Import data and validate
    forecast_production = DataHandler("Forecasts_Hour.csv", sep=';', decimal=',')
    forecast_production_DK1_onshore = DataHandler()
    forecast_production_DK1_offshore = DataHandler()
    forecast_production_DK1_solar = DataHandler()
    forecast_production_DK2_onshore = DataHandler()
    forecast_production_DK2_offshore = DataHandler()
    forecast_production_DK2_solar = DataHandler()

    # Transform data: Drop 'HourDK' column, handle missing values, convert MTU to datetime
    forecast_production = forecast_production.transform_data(
        drop_columns=['HourDK', 'Forecast Intraday', 'Forecast5Hour', 'Forecast1Hour',  'ForecastCurrent', 'TimestampUTC', 'TimestampDK'])
    for col in forecast_production.data.columns:
        for col in forecast_production.data.columns:
            if col not in ["HourUTC", "PriceArea", "ForecastType"]:
                forecast_production.data[col] = pd.to_numeric(forecast_production.data[col], errors='coerce')
    forecast_production.data["HourUTC"] = (
        forecast_production.data["HourUTC"]
        .pipe(pd.to_datetime, format='%Y-%m-%d %H:%M:%S', utc=True)
    )
    forecast_production.set_data(
        forecast_production.data[
            forecast_production.data["HourUTC"] <= pd.Timestamp("2025-09-22", tz="UTC")
        ]
    )

    # Split data into DK1 and DK2 and by ForecastType
    forecast_production_DK1_onshore.set_data(
        forecast_production.data[
            (forecast_production.data["PriceArea"] == "DK1") &
            (forecast_production.data["ForecastType"] == "Onshore Wind")
        ].drop(columns=["PriceArea", "ForecastType"])
    )
    forecast_production_DK1_offshore.set_data(
        forecast_production.data[
            (forecast_production.data["PriceArea"] == "DK1") &
            (forecast_production.data["ForecastType"] == "Offshore Wind")
        ].drop(columns=["PriceArea", "ForecastType"])
    )
    forecast_production_DK1_solar.set_data(
        forecast_production.data[
            (forecast_production.data["PriceArea"] == "DK1") &
            (forecast_production.data["ForecastType"] == "Solar")
        ].drop(columns=["PriceArea", "ForecastType"])
    )
    forecast_production_DK2_onshore.set_data(
        forecast_production.data[
            (forecast_production.data["PriceArea"] == "DK2") &
            (forecast_production.data["ForecastType"] == "Onshore Wind")
        ].drop(columns=["PriceArea", "ForecastType"])
    )
    forecast_production_DK2_offshore.set_data(
        forecast_production.data[
            (forecast_production.data["PriceArea"] == "DK2") &
            (forecast_production.data["ForecastType"] == "Offshore Wind")
        ].drop(columns=["PriceArea", "ForecastType"])
    )
    forecast_production_DK2_solar.set_data(
        forecast_production.data[
            (forecast_production.data["PriceArea"] == "DK2") &
            (forecast_production.data["ForecastType"] == "Solar")
        ].drop(columns=["PriceArea", "ForecastType"])
    )

    # DK1 onshore wind data processing
    # Check for duplicate HourUTC values
    datetime_duplicates = forecast_production_DK1_onshore.data["HourUTC"].duplicated().sum()
    if datetime_duplicates > 0:
        raise ValueError(f"Duplicate HourUTC values found: {datetime_duplicates}")

    # Change column names to indicate DK1
    for col in forecast_production_DK1_onshore.data.columns:
        if col != "HourUTC":
            forecast_production_DK1_onshore.data.rename(columns={col: f"{col}_DK1_onshore"}, inplace=True)

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        forecast_production_DK1_onshore = forecast_production_DK1_onshore.transform_data(
        drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        forecast_production_DK1_onshore.set_data(forecast_production_DK1_onshore.data
                                    .set_index("HourUTC").resample("h")
                                       .sum(min_count=1))
        # Reset index to a column
        forecast_production_DK1_onshore.set_data(forecast_production_DK1_onshore.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            forecast_production_DK1_onshore.data, left_on='datetime',
            right_on='HourUTC', how='left').drop(columns=['HourUTC'])
    else:
        print("[WARNING] Duplicate HourUTC values found. "
              "Please check the data for inconsistencies.")

    print(forecast_production_DK1_onshore.info())
    del forecast_production_DK1_onshore


    # DK1 offshore wind data processing
    # Check for duplicate HourUTC values
    datetime_duplicates = forecast_production_DK1_offshore.data["HourUTC"].duplicated().sum()
    if datetime_duplicates > 0:
        raise ValueError(f"Duplicate HourUTC values found: {datetime_duplicates}")

    # Change column names to indicate DK1
    for col in forecast_production_DK1_offshore.data.columns:
        if col != "HourUTC":
            forecast_production_DK1_offshore.data.rename(columns={col: f"{col}_DK1_offshore"}, inplace=True)

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        forecast_production_DK1_offshore = forecast_production_DK1_offshore.transform_data(
        drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        forecast_production_DK1_offshore.set_data(forecast_production_DK1_offshore.data
                                    .set_index("HourUTC").resample("h")
                                       .sum(min_count=1))
        # Reset index to a column
        forecast_production_DK1_offshore.set_data(forecast_production_DK1_offshore.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            forecast_production_DK1_offshore.data, left_on='datetime',
            right_on='HourUTC', how='left').drop(columns=['HourUTC'])
    else:
        print("[WARNING] Duplicate HourUTC values found. "
              "Please check the data for inconsistencies.")

    print(forecast_production_DK1_offshore.info())
    del forecast_production_DK1_offshore

    # DK1 solar data processing
    # Check for duplicate HourUTC values
    datetime_duplicates = forecast_production_DK1_solar.data["HourUTC"].duplicated().sum()
    if datetime_duplicates > 0:
        raise ValueError(f"Duplicate HourUTC values found: {datetime_duplicates}")

    # Change column names to indicate DK1
    for col in forecast_production_DK1_solar.data.columns:
        if col != "HourUTC":
            forecast_production_DK1_solar.data.rename(columns={col: f"{col}_DK1_solar"}, inplace=True)

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        forecast_production_DK1_solar = forecast_production_DK1_solar.transform_data(
        drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        forecast_production_DK1_solar.set_data(forecast_production_DK1_solar.data
                                    .set_index("HourUTC").resample("h")
                                       .sum(min_count=1))
        # Reset index to a column
        forecast_production_DK1_solar.set_data(forecast_production_DK1_solar.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            forecast_production_DK1_solar.data, left_on='datetime',
            right_on='HourUTC', how='left').drop(columns=['HourUTC'])
    else:
        print("[WARNING] Duplicate HourUTC values found. "
              "Please check the data for inconsistencies.")

    forecast_production_DK1_solar.info()
    del forecast_production_DK1_solar

    # DK2 onshore wind data processing
    # Check for duplicate HourUTC values
    datetime_duplicates = forecast_production_DK2_onshore.data["HourUTC"].duplicated().sum()
    if datetime_duplicates > 0:
        raise ValueError(f"Duplicate HourUTC values found: {datetime_duplicates}")

    # Change column names to indicate DK2
    for col in forecast_production_DK2_onshore.data.columns:
        if col != "HourUTC":
            forecast_production_DK2_onshore.data.rename(columns={col: f"{col}_DK2_onshore"}, inplace=True)

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        forecast_production_DK2_onshore = forecast_production_DK2_onshore.transform_data(
        drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        forecast_production_DK2_onshore.set_data(forecast_production_DK2_onshore.data
                                    .set_index("HourUTC").resample("h")
                                       .sum(min_count=1))
        # Reset index to a column
        forecast_production_DK2_onshore.set_data(forecast_production_DK2_onshore.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            forecast_production_DK2_onshore.data, left_on='datetime',
            right_on='HourUTC', how='left').drop(columns=['HourUTC'])
    else:
        print("[WARNING] Duplicate HourUTC values found. "
              "Please check the data for inconsistencies.")

    print(forecast_production_DK2_onshore.info())
    del forecast_production_DK2_onshore

    # DK2 offshore wind data processing
    # Check for duplicate HourUTC values
    datetime_duplicates = forecast_production_DK2_offshore.data["HourUTC"].duplicated().sum()
    if datetime_duplicates > 0:
        raise ValueError(f"Duplicate HourUTC values found: {datetime_duplicates}")

    # Change column names to indicate DK2
    for col in forecast_production_DK2_offshore.data.columns:
        if col != "HourUTC":
            forecast_production_DK2_offshore.data.rename(columns={col: f"{col}_DK2_offshore"}, inplace=True)

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        forecast_production_DK2_offshore = forecast_production_DK2_offshore.transform_data(
        drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        forecast_production_DK2_offshore.set_data(forecast_production_DK2_offshore.data
                                    .set_index("HourUTC").resample("h")
                                       .sum(min_count=1))
        # Reset index to a column
        forecast_production_DK2_offshore.set_data(forecast_production_DK2_offshore.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            forecast_production_DK2_offshore.data, left_on='datetime',
            right_on='HourUTC', how='left').drop(columns=['HourUTC'])
    else:
        print("[WARNING] Duplicate HourUTC values found. "
              "Please check the data for inconsistencies.")

    print(forecast_production_DK2_offshore.info())
    del forecast_production_DK2_offshore

    # DK2 solar data processing
    # Check for duplicate HourUTC values
    datetime_duplicates = forecast_production_DK2_solar.data["HourUTC"].duplicated().sum()
    if datetime_duplicates > 0:
        raise ValueError(f"Duplicate HourUTC values found: {datetime_duplicates}")

    # Change column names to indicate DK2
    for col in forecast_production_DK2_solar.data.columns:
        if col != "HourUTC":
            forecast_production_DK2_solar.data.rename(columns={col: f"{col}_DK2_solar"}, inplace=True)

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        forecast_production_DK2_solar = forecast_production_DK2_solar.transform_data(
        drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        forecast_production_DK2_solar.set_data(forecast_production_DK2_solar.data
                                    .set_index("HourUTC").resample("h")
                                       .sum(min_count=1))
        # Reset index to a column
        forecast_production_DK2_solar.set_data(forecast_production_DK2_solar.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            forecast_production_DK2_solar.data, left_on='datetime',
            right_on='HourUTC', how='left').drop(columns=['HourUTC'])
    else:
        print("[WARNING] Duplicate HourUTC values found. "
              "Please check the data for inconsistencies.")

    print(forecast_production_DK2_solar.info())
    del forecast_production_DK2_solar
    print("Shape features: ", features_df.shape)

###########################################################################

    print("=" * 60)
    print("Importing Data: Generation Forecast - Day ahead - DK1")
    print("=" * 60)

    # Import data and validate
    forecast_generation_DK1 = DataHandler()

    forecast_generation_2023_DK1 = DataHandler(
        'Generation Forecast - Day ahead_2023 - DK1.csv')
    forecast_generation_2024_DK1 = DataHandler(
        'Generation Forecast - Day ahead_2024 - DK1.csv')
    forecast_generation_2025_DK1 = DataHandler(
        'Generation Forecast - Day ahead_2025 - DK1.csv')

    forecast_generation_DK1.set_data(
        pd.concat([
            forecast_generation_2023_DK1.data,
            forecast_generation_2024_DK1.data,
            forecast_generation_2025_DK1.data
        ], ignore_index=True)
    )

    del forecast_generation_2023_DK1, forecast_generation_2024_DK1, forecast_generation_2025_DK1

    # Transform data: Drop 'Area' column, handle missing values, convert MTU to datetime
    forecast_generation_DK1 = forecast_generation_DK1.transform_data(
        drop_columns=['Area','Scheduled Consumption [MW] (D) - BZN|DK1'])
    for col in forecast_generation_DK1.data.columns:
        if col != "MTU":
            forecast_generation_DK1.data[col] = pd.to_numeric(forecast_generation_DK1.data[col], errors='coerce')
            forecast_generation_DK1.data.rename(columns={col: f"{col}_DK1"}, inplace=True)
    forecast_generation_DK1.data["MTU"] = (
        forecast_generation_DK1.data["MTU"]
        .replace(" (UTC)", "", regex=False).str.split(" - ").str[0].
        pipe(pd.to_datetime, format='%d.%m.%Y %H:%M', utc=True)
    )
    forecast_generation_DK1.set_data(
        forecast_generation_DK1.data[
            forecast_generation_DK1.data["MTU"] <= pd.Timestamp("2025-09-22", tz="UTC")
        ]
    )
    # Check for duplicate MTU values
    datetime_duplicates = forecast_generation_DK1.data["MTU"].duplicated().sum()
    print(f"Number of duplicate MTU values: {datetime_duplicates}")

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        forecast_generation_DK1 = forecast_generation_DK1.transform_data(
        drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        forecast_generation_DK1.set_data(forecast_generation_DK1.data
                                       .set_index("MTU").resample("h")
                                       .sum(min_count=1))
        # Reset index to a column
        forecast_generation_DK1.set_data(forecast_generation_DK1.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            forecast_generation_DK1.data, left_on='datetime',
            right_on='MTU', how='left').drop(columns=['MTU'])
    else:
        print("[WARNING] Duplicate MTU values found. "
              "Please check the data for inconsistencies.")

    print(forecast_generation_DK1.info())
    del forecast_generation_DK1

###########################################################################

    print("=" * 60)
    print("Importing Data: Generation Forecast - Day ahead - DK2")
    print("=" * 60)

    # Import data and validate
    forecast_generation_DK2 = DataHandler()

    forecast_generation_2023_DK2 = DataHandler(
        'Generation Forecast - Day ahead_2023 - DK2.csv')
    forecast_generation_2024_DK2 = DataHandler(
        'Generation Forecast - Day ahead_2024 - DK2.csv')
    forecast_generation_2025_DK2 = DataHandler(
        'Generation Forecast - Day ahead_2025 - DK2.csv')

    forecast_generation_DK2.set_data(
        pd.concat([
            forecast_generation_2023_DK2.data,
            forecast_generation_2024_DK2.data,
            forecast_generation_2025_DK2.data
        ], ignore_index=True)
    )

    del forecast_generation_2023_DK2, forecast_generation_2024_DK2, forecast_generation_2025_DK2

    # Transform data: Drop 'Area' column, handle missing values, convert MTU to datetime
    forecast_generation_DK2 = forecast_generation_DK2.transform_data(
        drop_columns=['Area','Scheduled Consumption [MW] (D) - BZN|DK2'])
    for col in forecast_generation_DK2.data.columns:
        if col != "MTU":
            forecast_generation_DK2.data[col] = pd.to_numeric(forecast_generation_DK2.data[col], errors='coerce')
            forecast_generation_DK2.data.rename(columns={col: f"{col}_DK2"}, inplace=True)
    forecast_generation_DK2.data["MTU"] = (
        forecast_generation_DK2.data["MTU"]
        .replace(" (UTC)", "", regex=False).str.split(" - ").str[0].
        pipe(pd.to_datetime, format='%d.%m.%Y %H:%M', utc=True)
    )
    forecast_generation_DK2.set_data(
        forecast_generation_DK2.data[
            forecast_generation_DK2.data["MTU"] <= pd.Timestamp("2025-09-22", tz="UTC")
        ]
    )
    # Check for duplicate MTU values
    datetime_duplicates = forecast_generation_DK2.data["MTU"].duplicated().sum()
    print(f"Number of duplicate MTU values: {datetime_duplicates}")

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        forecast_generation_DK2 = forecast_generation_DK2.transform_data(
        drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        forecast_generation_DK2.set_data(forecast_generation_DK2.data
                                       .set_index("MTU").resample("h")
                                       .sum(min_count=1))
        # Reset index to a column
        forecast_generation_DK2.set_data(forecast_generation_DK2.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            forecast_generation_DK2.data, left_on='datetime',
            right_on='MTU', how='left').drop(columns=['MTU'])
    else:
        print("[WARNING] Duplicate MTU values found. "
              "Please check the data for inconsistencies.")

    print(forecast_generation_DK2.info())
    del forecast_generation_DK2
    print("Shape features: ", features_df.shape)

    ###########################################################################

    print("=" * 60)
    print("Importing Data: Generation Forecast - Day ahead - DK1")
    print("=" * 60)

    # Import data and validate
    forecast_generation_DK1 = DataHandler()

    forecast_generation_2023_DK1 = DataHandler(
        'Generation Forecasts for Wind and Solar_2023 - DK1.csv')
    forecast_generation_2024_DK1 = DataHandler(
        'Generation Forecasts for Wind and Solar_2024 - DK1.csv')
    forecast_generation_2025_DK1 = DataHandler(
        'Generation Forecasts for Wind and Solar_2025 - DK1.csv')

    forecast_generation_DK1.set_data(
        pd.concat([
            forecast_generation_2023_DK1.data,
            forecast_generation_2024_DK1.data,
            forecast_generation_2025_DK1.data
        ], ignore_index=True)
    )

    del forecast_generation_2023_DK1, forecast_generation_2024_DK1, forecast_generation_2025_DK1

    # Transform data: Drop 'Area' column, handle missing values, convert MTU to datetime
    forecast_generation_DK1 = forecast_generation_DK1.transform_data(
        drop_columns=['Generation - Solar [MW] Intraday / BZN|DK1',
                      'Generation - Solar [MW] Current / BZN|DK1',
                      'Generation - Wind Offshore [MW] Intraday / BZN|DK1',
                      'Generation - Wind Offshore [MW] Current / BZN|DK1',
                      'Generation - Wind Onshore [MW] Intraday / BZN|DK1',
                      'Generation - Wind Onshore [MW] Current / BZN|DK1'])
    for col in forecast_generation_DK1.data.columns:
        if col != "MTU (UTC)":
            forecast_generation_DK1.data[col] = pd.to_numeric(forecast_generation_DK1.data[col], errors='coerce')
            forecast_generation_DK1.data.rename(columns={col: f"{col}_DK1"}, inplace=True)
    forecast_generation_DK1.data["MTU (UTC)"] = (
        forecast_generation_DK1.data["MTU (UTC)"]
        .str.split(" - ").str[0].
        pipe(pd.to_datetime, format='%d.%m.%Y %H:%M', utc=True)
    )
    forecast_generation_DK1.set_data(
        forecast_generation_DK1.data[
            forecast_generation_DK1.data["MTU (UTC)"] <= pd.Timestamp("2025-09-22", tz="UTC")
        ]
    )
    # Check for duplicate MTU values
    datetime_duplicates = forecast_generation_DK1.data["MTU (UTC)"].duplicated().sum()
    print(f"Number of duplicate MTU values: {datetime_duplicates}")

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        forecast_generation_DK1 = forecast_generation_DK1.transform_data(
        drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        forecast_generation_DK1.set_data(forecast_generation_DK1.data
                                       .set_index("MTU (UTC)").resample("h")
                                       .sum(min_count=1))
        # Reset index to a column
        forecast_generation_DK1.set_data(forecast_generation_DK1.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            forecast_generation_DK1.data, left_on='datetime',
            right_on='MTU (UTC)', how='left').drop(columns=['MTU (UTC)'])
    else:
        print("[WARNING] Duplicate MTU values found. "
              "Please check the data for inconsistencies.")

    print(forecast_generation_DK1.info())
    del forecast_generation_DK1

    ###########################################################################

    print("=" * 60)
    print("Importing Data: Generation Forecast - Day ahead - DK2")
    print("=" * 60)

    # Import data and validate
    forecast_generation_DK2 = DataHandler()

    forecast_generation_2023_DK2 = DataHandler(
        'Generation Forecasts for Wind and Solar_2023 - DK2.csv')
    forecast_generation_2024_DK2 = DataHandler(
        'Generation Forecasts for Wind and Solar_2024 - DK2.csv')
    forecast_generation_2025_DK2 = DataHandler(
        'Generation Forecasts for Wind and Solar_2025 - DK2.csv')

    forecast_generation_DK2.set_data(
        pd.concat([
            forecast_generation_2023_DK2.data,
            forecast_generation_2024_DK2.data,
            forecast_generation_2025_DK2.data
        ], ignore_index=True)
    )

    del forecast_generation_2023_DK2, forecast_generation_2024_DK2, forecast_generation_2025_DK2

    # Transform data: Drop 'Area' column, handle missing values, convert MTU to datetime
    forecast_generation_DK2 = forecast_generation_DK2.transform_data(
        drop_columns=['Generation - Solar [MW] Intraday / BZN|DK2',
                      'Generation - Solar [MW] Current / BZN|DK2',
                      'Generation - Wind Offshore [MW] Intraday / BZN|DK2',
                      'Generation - Wind Offshore [MW] Current / BZN|DK2',
                      'Generation - Wind Onshore [MW] Intraday / BZN|DK2',
                      'Generation - Wind Onshore [MW] Current / BZN|DK2'])
    for col in forecast_generation_DK2.data.columns:
        if col != "MTU (UTC)":
            forecast_generation_DK2.data[col] = pd.to_numeric(forecast_generation_DK2.data[col], errors='coerce')
            forecast_generation_DK2.data.rename(columns={col: f"{col}_DK2"}, inplace=True)
    forecast_generation_DK2.data["MTU (UTC)"] = (
        forecast_generation_DK2.data["MTU (UTC)"]
        .str.split(" - ").str[0].
        pipe(pd.to_datetime, format='%d.%m.%Y %H:%M', utc=True)
    )
    forecast_generation_DK2.set_data(
        forecast_generation_DK2.data[
            forecast_generation_DK2.data["MTU (UTC)"] <= pd.Timestamp("2025-09-22", tz="UTC")
        ]
    )
    # Check for duplicate MTU values
    datetime_duplicates = forecast_generation_DK2.data["MTU (UTC)"].duplicated().sum()
    print(f"Number of duplicate MTU values: {datetime_duplicates}")

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        forecast_generation_DK2 = forecast_generation_DK2.transform_data(
        drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        forecast_generation_DK2.set_data(forecast_generation_DK2.data
                                       .set_index("MTU (UTC)").resample("h")
                                       .sum(min_count=1))
        # Reset index to a column
        forecast_generation_DK2.set_data(forecast_generation_DK2.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            forecast_generation_DK2.data, left_on='datetime',
            right_on='MTU (UTC)', how='left').drop(columns=['MTU (UTC)'])
    else:
        print("[WARNING] Duplicate MTU values found. "
              "Please check the data for inconsistencies.")

    print(forecast_generation_DK2.info())
    del forecast_generation_DK2
    print("Shape features: ", features_df.shape)

    ###########################################################################

    print("=" * 60)
    print("Importing Data: mFRRCapacityMarket")
    print("=" * 60)

    # Import data and validate
    mFRR_capacity_market = DataHandler("mFRRCapacityMarket.csv", sep=';', decimal=',')
    mFRR_reserves_DK1_2023 = DataHandler("MfrrReservesDK1.csv", sep=';', decimal=',')
    mFRR_reserves_DK2_2023 = DataHandler("MfrrReservesDK2.csv", sep=';', decimal=',')
    mFRR_capacity_market_DK1 = DataHandler()
    mFRR_capacity_market_DK2 = DataHandler()

    # Drop columns
    mFRR_capacity_market = mFRR_capacity_market.transform_data(
        drop_columns=['HourDK', 'mFRR_DownPriceDKK', 'mFRR_UpPriceDKK'])
    mFRR_reserves_DK1_2023 = mFRR_reserves_DK1_2023.transform_data(
        drop_columns=['HourDK', 'mFRR_DownExpected', 'mFRR_DownPriceDKK',
                      'mFRR_DownExpectedXtra', 'mFRR_DownPurchasedXtra',
                      'mFRR_DownPriceXtraDKK', 'mFRR_DownPriceXtraEUR',
                      'mFRR_UpExpected', 'mFRR_UpPriceDKK',
                      'mFRR_UpExpectedXtra', 'mFRR_UpPurchasedXtra',
                      'mFRR_UpPriceXtraDKK', 'mFRR_UpPriceXtraEUR'])
    mFRR_reserves_DK2_2023 = mFRR_reserves_DK2_2023.transform_data(
        drop_columns=['HourDK', 'mFRR_DownExpected', 'mFRR_DownPriceDKK',
                        'mFRR_UpExpected', 'mFRR_UpPriceDKK'])

    # Split mFRR capacity market data into DK1 and DK2
    mFRR_capacity_market_DK1.set_data(
        mFRR_capacity_market.data[
            mFRR_capacity_market.data["PriceArea"] == "DK1"
        ].drop(columns=["PriceArea"])
    )
    mFRR_capacity_market_DK2.set_data(
        mFRR_capacity_market.data[
            mFRR_capacity_market.data["PriceArea"] == "DK2"
        ].drop(columns=["PriceArea"])
    )
    del mFRR_capacity_market

    # Combine mFRR reserves data for DK1 and DK2
    mFRR_capacity_market_DK1.set_data(
        pd.concat([mFRR_reserves_DK1_2023.data, mFRR_capacity_market_DK1.data],
                  ignore_index=True)
    )
    mFRR_capacity_market_DK2.set_data(
        pd.concat([mFRR_reserves_DK2_2023.data, mFRR_capacity_market_DK2.data],
                  ignore_index=True)
    )

    del mFRR_reserves_DK1_2023, mFRR_reserves_DK2_2023

    # DK1 data processing
    # Transform data: Handle missing values, convert MTU to datetime
    for col in mFRR_capacity_market_DK1.data.columns:
        if col not in ["HourUTC"]:
            mFRR_capacity_market_DK1.data[col] = pd.to_numeric(mFRR_capacity_market_DK1.data[col], errors='coerce')
    mFRR_capacity_market_DK1.data["HourUTC"] = (
        mFRR_capacity_market_DK1.data["HourUTC"]
        .pipe(pd.to_datetime, format='%Y-%m-%d %H:%M:%S', utc=True)
    )
    mFRR_capacity_market_DK1.set_data(
        mFRR_capacity_market_DK1.data[
            mFRR_capacity_market_DK1.data["HourUTC"] <= pd.Timestamp("2025-09-22", tz="UTC")
        ]
    )

    # Check for duplicate HourUTC values
    datetime_duplicates = mFRR_capacity_market_DK1.data["HourUTC"].duplicated().sum()
    if datetime_duplicates > 0:
        raise ValueError(f"Duplicate HourUTC values found: {datetime_duplicates}")

    # Change column names to indicate DK1
    for col in mFRR_capacity_market_DK1.data.columns:
        if col != "HourUTC":
            mFRR_capacity_market_DK1.data.rename(columns={col: f"{col}_DK1"}, inplace=True)

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        mFRR_capacity_market_DK1 = mFRR_capacity_market_DK1.transform_data(
        drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        mFRR_capacity_market_DK1.set_data(mFRR_capacity_market_DK1.data
                                            .set_index("HourUTC").resample("h")
                                       .sum(min_count=1))
        # Reset index to a column
        mFRR_capacity_market_DK1.set_data(mFRR_capacity_market_DK1.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            mFRR_capacity_market_DK1.data, left_on='datetime',
            right_on='HourUTC', how='left').drop(columns=['HourUTC'])
    else:
        print("[WARNING] Duplicate HourUTC values found. "
              "Please check the data for inconsistencies.")

    print(mFRR_capacity_market_DK1.info())
    del mFRR_capacity_market_DK1

    # DK2 data processing
    # Transform data: Handle missing values, convert MTU to datetime
    for col in mFRR_capacity_market_DK2.data.columns:
        for col in mFRR_capacity_market_DK2.data.columns:
            if col not in ["HourUTC"]:
                mFRR_capacity_market_DK2.data[col] = pd.to_numeric(mFRR_capacity_market_DK2.data[col], errors='coerce')
    mFRR_capacity_market_DK2.data["HourUTC"] = (
        mFRR_capacity_market_DK2.data["HourUTC"]
        .pipe(pd.to_datetime, format='%Y-%m-%d %H:%M:%S', utc=True)
    )
    mFRR_capacity_market_DK2.set_data(
        mFRR_capacity_market_DK2.data[
            mFRR_capacity_market_DK2.data["HourUTC"] <= pd.Timestamp("2025-09-22", tz="UTC")
        ]
    )

    # Check for duplicate HourUTC values
    datetime_duplicates = mFRR_capacity_market_DK2.data["HourUTC"].duplicated().sum()
    if datetime_duplicates > 0:
        raise ValueError(f"Duplicate HourUTC values found: {datetime_duplicates}")

    # Change column names to indicate DK2
    for col in mFRR_capacity_market_DK2.data.columns:
        if col != "HourUTC":
            mFRR_capacity_market_DK2.data.rename(columns={col: f"{col}_DK2"}, inplace=True)

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        mFRR_capacity_market_DK2 = mFRR_capacity_market_DK2.transform_data(
        drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        mFRR_capacity_market_DK2.set_data(mFRR_capacity_market_DK2.data
                                    .set_index("HourUTC").resample("h")
                                       .sum(min_count=1))
        # Reset index to a column
        mFRR_capacity_market_DK2.set_data(mFRR_capacity_market_DK2.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            mFRR_capacity_market_DK2.data, left_on='datetime',
            right_on='HourUTC', how='left').drop(columns=['HourUTC'])
    else:
        print("[WARNING] Duplicate HourUTC values found. "
              "Please check the data for inconsistencies.")

    print(mFRR_capacity_market_DK2.info())
    del mFRR_capacity_market_DK2
    print("Shape features: ", features_df.shape)

    ###########################################################################

    print("=" * 60)
    print("Importing Data: ProductionConsumptionSettlement")
    print("=" * 60)

    # Import data and validate
    actual_production_consumption = DataHandler("ProductionConsumptionSettlement.csv",
                                 sep=';', decimal=',')
    actual_production_consumption_DK1 = DataHandler()
    actual_production_consumption_DK2 = DataHandler()

    # Transform data: Drop 'HourDK' column, handle missing values, convert MTU to datetime
    actual_production_consumption = actual_production_consumption.transform_data(
        drop_columns=['HourDK'])
    for col in actual_production_consumption.data.columns:
        for col in actual_production_consumption.data.columns:
            if col not in ["HourUTC", "PriceArea"]:
                actual_production_consumption.data[col] = pd.to_numeric(actual_production_consumption.data[col], errors='coerce')
    actual_production_consumption.data["HourUTC"] = (
        actual_production_consumption.data["HourUTC"]
        .pipe(pd.to_datetime, format='%Y-%m-%d %H:%M:%S', utc=True)
    )
    actual_production_consumption.set_data(
        actual_production_consumption.data[
            actual_production_consumption.data["HourUTC"] <= pd.Timestamp("2025-09-22", tz="UTC")
        ]
    )

    # Split data into DK1 and DK2
    actual_production_consumption_DK1.set_data(
        actual_production_consumption.data[actual_production_consumption.data["PriceArea"] == "DK1"]
        .drop(columns=["PriceArea"])
    )

    actual_production_consumption_DK2.set_data(
        actual_production_consumption.data[actual_production_consumption.data["PriceArea"] == "DK2"]
        .drop(columns=["PriceArea"])
    )

    del actual_production_consumption

    # DK1 data processing
    # Check for duplicate HourUTC values
    datetime_duplicates = actual_production_consumption_DK1.data["HourUTC"].duplicated().sum()
    if datetime_duplicates > 0:
        raise ValueError(f"Duplicate HourUTC values found: {datetime_duplicates}")

    # Change column names to indicate DK1
    for col in actual_production_consumption_DK1.data.columns:
        if col != "HourUTC":
            actual_production_consumption_DK1.data.rename(columns={col: f"{col}_DK1"}, inplace=True)

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        actual_production_consumption_DK1 = actual_production_consumption_DK1.transform_data(
        drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        actual_production_consumption_DK1.set_data(actual_production_consumption_DK1.data
                                                   .set_index("HourUTC").resample("h")
                                                   .sum(min_count=1))

        # Create lagged features (48 hours)
        actual_production_consumption_DK1.set_data(actual_production_consumption_DK1.data.shift(48))
        actual_production_consumption_DK1.data.columns = [f"{col}_Lag48h" for col in actual_production_consumption_DK1.data.columns]

        # Reset index to a column
        actual_production_consumption_DK1.set_data(
            actual_production_consumption_DK1.data.reset_index()
            )

        # Merge with features_df
        features_df = features_df.merge(
            actual_production_consumption_DK1.data, left_on='datetime',
            right_on='HourUTC', how='left').drop(columns=['HourUTC'])
    else:
        print("[WARNING] Duplicate HourUTC values found. "
              "Please check the data for inconsistencies.")

    print(actual_production_consumption_DK1.info())
    del actual_production_consumption_DK1

    # DK2 data processing
    # Check for duplicate HourUTC values
    datetime_duplicates = actual_production_consumption_DK2.data["HourUTC"].duplicated().sum()
    if datetime_duplicates > 0:
        raise ValueError(f"Duplicate HourUTC values found: {datetime_duplicates}")

    # Change column names to indicate DK2
    for col in actual_production_consumption_DK2.data.columns:
        if col != "HourUTC":
            actual_production_consumption_DK2.data.rename(columns={col: f"{col}_DK2"}, inplace=True)

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        actual_production_consumption_DK2 = actual_production_consumption_DK2.transform_data(
        drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        actual_production_consumption_DK2.set_data(actual_production_consumption_DK2.data
                                    .set_index("HourUTC").resample("h")
                                       .sum(min_count=1))

        # Create lagged features (48 hours)
        actual_production_consumption_DK2.set_data(actual_production_consumption_DK2.data.shift(48))
        actual_production_consumption_DK2.data.columns = [f"{col}_Lag48h" for col in actual_production_consumption_DK2.data.columns]

        # Reset index to a column
        actual_production_consumption_DK2.set_data(actual_production_consumption_DK2.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            actual_production_consumption_DK2.data, left_on='datetime',
            right_on='HourUTC', how='left').drop(columns=['HourUTC'])
    else:
        print(" [WARNING] Duplicate HourUTC values found. "
              "Please check the data for inconsistencies.")

    print(actual_production_consumption_DK2.info())
    del actual_production_consumption_DK2
    print("Shape features: ", features_df.shape)

    ###########################################################################

    print("=" * 60)
    print("Importing Data: Total Load - Day Ahead _ Actual - DK1")
    print("=" * 60)

    # Import data and validate
    load_data_DK1 = DataHandler()

    load_data_DK1_2023 = DataHandler(
        'Total Load - Day Ahead _ Actual_2023 - DK1.csv')
    load_data_DK1_2024 = DataHandler(
        'Total Load - Day Ahead _ Actual_2024 - DK1.csv')
    load_data_DK1_2025 = DataHandler(
        'Total Load - Day Ahead _ Actual_2025 - DK1.csv')

    load_data_DK1.set_data(
        pd.concat([
            load_data_DK1_2023.data,
            load_data_DK1_2024.data,
            load_data_DK1_2025.data
        ], ignore_index=True)
    )

    del load_data_DK1_2023, load_data_DK1_2024, load_data_DK1_2025

    # Transform data: Drop 'Area' column, handle missing values
    for col in load_data_DK1.data.columns:
        if col != "Time (UTC)":
            load_data_DK1.data[col] = pd.to_numeric(load_data_DK1.data[col],
                                                    errors='coerce')
    load_data_DK1.data["Time (UTC)"] = (
        load_data_DK1.data["Time (UTC)"]
        .str.split(" - ").str[0].
        pipe(pd.to_datetime, format='%d.%m.%Y %H:%M', utc=True)
    )
    load_data_DK1.set_data(
        load_data_DK1.data[
            load_data_DK1.data["Time (UTC)"] <= pd.Timestamp("2025-09-22",
                                                             tz="UTC")
        ]
    )
    # Check for duplicate Time (UTC) values
    datetime_duplicates = load_data_DK1.data["Time (UTC)"].duplicated().sum()
    print(f"Number of duplicate Time (UTC) values: {datetime_duplicates}")

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        load_data_DK1 = load_data_DK1.transform_data(
            drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        load_data_DK1.set_data(load_data_DK1.data
                               .set_index("Time (UTC)").resample("h")
                               .sum(min_count=1))

        # Create lagged features (48 hours)
        load_data_DK1.data["Actual Total Load [MW] - BZN|DK1_Lag48h"] = load_data_DK1.data["Actual Total Load [MW] - BZN|DK1"].shift(48)
        load_data_DK1 = load_data_DK1.transform_data(drop_columns=["Actual Total Load [MW] - BZN|DK1"])

        # Reset index to a column
        load_data_DK1.set_data(load_data_DK1.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            load_data_DK1.data, left_on='datetime',
            right_on='Time (UTC)', how='left').drop(columns=['Time (UTC)'])
    else:
        print(" [WARNING] Duplicate Time (UTC) values found. "
              "Please check the data for inconsistencies.")

    load_data_DK1.info()
    del load_data_DK1

    ###########################################################################

    print("=" * 60)
    print("Importing Data: Total Load - Day Ahead _ Actual - DK2")
    print("=" * 60)

    # Import data and validate
    load_data_DK2 = DataHandler()

    load_data_DK2_2023 = DataHandler(
        'Total Load - Day Ahead _ Actual_2023 - DK2.csv')
    load_data_DK2_2024 = DataHandler(
        'Total Load - Day Ahead _ Actual_2024 - DK2.csv')
    load_data_DK2_2025 = DataHandler(
        'Total Load - Day Ahead _ Actual_2025 - DK2.csv')

    load_data_DK2.set_data(
        pd.concat([
            load_data_DK2_2023.data,
            load_data_DK2_2024.data,
            load_data_DK2_2025.data
        ], ignore_index=True)
    )

    del load_data_DK2_2023, load_data_DK2_2024, load_data_DK2_2025

    # Transform data: Drop 'Area' column, handle missing values
    for col in load_data_DK2.data.columns:
        if col != "Time (UTC)":
            load_data_DK2.data[col] = pd.to_numeric(load_data_DK2.data[col],
                                                    errors='coerce')
    load_data_DK2.data["Time (UTC)"] = (
        load_data_DK2.data["Time (UTC)"]
        .str.split(" - ").str[0].
        pipe(pd.to_datetime, format='%d.%m.%Y %H:%M', utc=True)
    )
    load_data_DK2.set_data(
        load_data_DK2.data[
            load_data_DK2.data["Time (UTC)"] <= pd.Timestamp("2025-09-22",
                                                             tz="UTC")
        ]
    )
    # Check for duplicate Time (UTC) values
    datetime_duplicates = load_data_DK2.data["Time (UTC)"].duplicated().sum()
    print(f"Number of duplicate Time (UTC) values: {datetime_duplicates}")

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        load_data_DK2 = load_data_DK2.transform_data(
            drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        load_data_DK2.set_data(load_data_DK2.data
                               .set_index("Time (UTC)").resample("h")
                               .sum(min_count=1))

        # Create lagged features (48 hours)
        load_data_DK2.data["Actual Total Load [MW] - BZN|DK2_Lag48h"] = load_data_DK2.data["Actual Total Load [MW] - BZN|DK2"].shift(48)
        load_data_DK2 = load_data_DK2.transform_data(drop_columns=["Actual Total Load [MW] - BZN|DK2"])

        # Reset index to a column
        load_data_DK2.set_data(load_data_DK2.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            load_data_DK2.data, left_on='datetime',
            right_on='Time (UTC)', how='left').drop(columns=['Time (UTC)'])
    else:
        print("[WARNING] Duplicate Time (UTC) values found. "
              "Please check the data for inconsistencies.")

    load_data_DK2.info()
    del load_data_DK2

    ###########################################################################

    print("=" * 60)
    print("Import target data: RegulatingBalancePowerdata")
    print("=" * 60)

    # Import data and validate
    imbalance = DataHandler("RegulatingBalancePowerdata.csv",
                            sep=';', decimal=',')
    imbalance_DK1 = DataHandler()
    imbalance_DK2 = DataHandler()

    # Transform data: Drop 'HourDK' column, handle missing values
    imbalance = imbalance.transform_data(
        drop_columns=['HourDK',
                      'mFRRUpActSpec',
                      'mFRRDownActSpec',
                      'ImbalanceMWh',
                      'ImbalancePriceDKK',
                      'BalancingPowerPriceUpDKK',
                      'BalancingPowerPriceDownDKK']
                      )
    for col in imbalance.data.columns:
        if col not in ["HourUTC", "PriceArea"]:
            imbalance.data[col] = pd.to_numeric(imbalance.data[col],
                                                errors='coerce')
    imbalance.data["HourUTC"] = (
        imbalance.data["HourUTC"]
        .pipe(pd.to_datetime, format='%Y-%m-%d %H:%M:%S', utc=True)
    )
    imbalance.set_data(
        imbalance.data[
            imbalance.data["HourUTC"] <= pd.Timestamp("2025-09-22", tz="UTC")
        ]
    )

    # add column with imbalance direction, 1 if surplus, -1 if deficit
    imbalance.data['ImbalanceDirection'] = (
        imbalance.data.apply(
            lambda row: 0 if (row['ImbalancePriceEUR'] == row['BalancingPowerPriceDownEUR'] and
                            row['ImbalancePriceEUR'] == row['BalancingPowerPriceUpEUR'])
            else 1 if row['ImbalancePriceEUR'] == row['BalancingPowerPriceDownEUR']
            else -1 if row['ImbalancePriceEUR'] == row['BalancingPowerPriceUpEUR']
            else 0,
            axis=1
        )
    )

    # add column with imbalance in MWh
    imbalance.data['ImbalanceMWh'] = (imbalance.data['mFRRDownActBal'] -
                                      imbalance.data['mFRRUpActBal'])

    # drop BalancingPowerPriceDownEUR and BalancingPowerPriceUpEUR
    imbalance = imbalance.transform_data(
        drop_columns=['BalancingPowerPriceDownEUR',
                      'BalancingPowerPriceUpEUR',
                      'mFRRUpActBal',
                      'mFRRDownActBal']
    )

    # Split data into DK1 and DK2
    imbalance_DK1.set_data(
        imbalance.data[imbalance.data["PriceArea"] == "DK1"]
        .drop(columns=["PriceArea"])
    )

    imbalance_DK2.set_data(
        imbalance.data[imbalance.data["PriceArea"] == "DK2"]
        .drop(columns=["PriceArea"])
    )

    del imbalance

    # DK1 data processing
    # Check for duplicate HourUTC values
    datetime_duplicates = imbalance_DK1.data["HourUTC"].duplicated().sum()
    if datetime_duplicates > 0:
        raise ValueError(
            f"Duplicate HourUTC values found: {datetime_duplicates}")

    # Change column names to indicate DK1
    for col in imbalance_DK1.data.columns:
        if col != "HourUTC":
            imbalance_DK1.data.rename(columns={col: f"{col}_DK1"},
                                      inplace=True)

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        imbalance_DK1 = imbalance_DK1.transform_data(
            drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        imbalance_DK1.set_data(imbalance_DK1.data
                               .set_index("HourUTC").resample("h")
                               .sum(min_count=1))

        # Create lagged features (48 hours)
        for col in imbalance_DK1.data.columns:
            imbalance_DK1.data[f"{col}_Lag48"] = imbalance_DK1.data[col].shift(48)

        # change type to category
        imbalance_DK1.data['ImbalanceDirection_DK1'] = imbalance_DK1.data['ImbalanceDirection_DK1'].astype('string')

        # Reset index to a column
        imbalance_DK1.set_data(imbalance_DK1.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            imbalance_DK1.data, left_on='datetime',
            right_on='HourUTC', how='left').drop(columns=['HourUTC'])
    else:
        print("[WARNING] Duplicate HourUTC values found. "
              "Please check the data for inconsistencies.")

    del imbalance_DK1

    # DK2 data processing
    # Check for duplicate HourUTC values
    datetime_duplicates = imbalance_DK2.data["HourUTC"].duplicated().sum()
    if datetime_duplicates > 0:
        raise ValueError(
            f"Duplicate HourUTC values found: {datetime_duplicates}")

    # Change column names to indicate DK2
    for col in imbalance_DK2.data.columns:
        if col != "HourUTC":
            imbalance_DK2.data.rename(columns={col: f"{col}_DK2"},
                                      inplace=True)

    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        imbalance_DK2 = imbalance_DK2.transform_data(
            drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        imbalance_DK2.set_data(imbalance_DK2.data
                               .set_index("HourUTC").resample("h")
                               .sum(min_count=1))

        # Create lagged features (48 hours)
        for col in imbalance_DK2.data.columns:
            imbalance_DK2.data[f"{col}_Lag48"] = imbalance_DK2.data[col].shift(48)

        # change type to category
        imbalance_DK2.data['ImbalanceDirection_DK2'] = imbalance_DK2.data['ImbalanceDirection_DK2'].astype('string')

        # Reset index to a column
        imbalance_DK2.set_data(imbalance_DK2.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            imbalance_DK2.data, left_on='datetime',
            right_on='HourUTC', how='left').drop(columns=['HourUTC'])
    else:
        print("WARNING: Duplicate HourUTC values found. "
              "Please check the data for inconsistencies.")

    imbalance_DK2.info()
    del imbalance_DK2

    ###########################################################################

    print("=" * 60)
    print("Import target data: Enfor_DA_wind_power_forecast")
    print("=" * 60)

    # Import data and validate
    wind_forecast = DataHandler("Enfor_DA_wind_power_forecast.csv")

    # Transform data: Drop 'HourDK' column, handle missing values
    wind_forecast = wind_forecast.transform_data(
        drop_columns=['Time_end',
                      'PTime',
                      'SCADAPowerMeas'],
        rename_columns={'Time_begin': 'HourUTC',
                        'PowerPred': 'WindFarm_WindPowerForecast',
                        'SettlementPowerMeas': 'WindFarm_ActualWindPower'}
                      )
    for col in wind_forecast.data.columns:
        if col not in ["HourUTC", "PTime"]:
            wind_forecast.data[col] = pd.to_numeric(wind_forecast.data[col],
                                                errors='coerce')

    wind_forecast.data["HourUTC"] = pd.to_datetime(
        wind_forecast.data["HourUTC"], format='%Y-%m-%d %H:%M:%S', utc=True)

    wind_forecast.set_data(
        wind_forecast.data[
            (wind_forecast.data["HourUTC"] >= pd.Timestamp("2023-01-01", tz="UTC")) &
            (wind_forecast.data["HourUTC"] <= pd.Timestamp("2025-09-22", tz="UTC"))
        ]
    )

    # Resolve duplicates (use latest update)
    wind_forecast.set_data(
        wind_forecast.data.sort_values('HourUTC').drop_duplicates(subset='HourUTC', keep='last')
    )

    # Check for duplicate HourUTC values
    datetime_duplicates = wind_forecast.data["HourUTC"].duplicated().sum()
    if datetime_duplicates > 0:
        raise ValueError(
            f"Duplicate HourUTC values found: {datetime_duplicates}")


    if datetime_duplicates == 0:
        # Drop columns with more than 20% missing values
        wind_forecast = wind_forecast.transform_data(
            drop_missing_threshold=0.1)

        # Resample to hourly frequency, summing values within each hour
        wind_forecast.set_data(wind_forecast.data
                               .set_index("HourUTC").resample("h")
                               .sum(min_count=1))

        # Create lagged features (48 hours)
        for col in wind_forecast.data.columns:

            wind_forecast.data[f"{col}_Lag48"] = wind_forecast.data[col].shift(48)

        # Reset index to a column
        wind_forecast.set_data(wind_forecast.data.reset_index())

        # Merge with features_df
        features_df = features_df.merge(
            wind_forecast.data, left_on='datetime',
            right_on='HourUTC', how='left').drop(columns=['HourUTC'])
    else:
        print("[WARNING] Duplicate HourUTC values found. "
              "Please check the data for inconsistencies.")

    wind_forecast.info()
    del wind_forecast


    ###########################################################################

    print("=" * 60)
    print("Adding time-based features")
    print("=" * 60)

    # Add hourly cyclical features
    features_df['sin_hour'] = np.sin(
        features_df['datetime'].dt.hour * (2 * np.pi / 24))
    features_df['cos_hour'] = np.cos(
        features_df['datetime'].dt.hour * (2 * np.pi / 24))

    # Add daily one-hot encoded features
    day_of_week = pd.get_dummies(features_df['datetime'].dt.dayofweek, prefix='day')
    features_df = pd.concat([features_df, day_of_week], axis=1)



    ##########################################################################

    print("=" * 60)
    print("Adding weather features")
    print("=" * 60)

    params = {
        "latitude": [57.5, 56.1, 54.8, 57.0, 56.1, 55.5, 56.2, 55.7, 55.4, 55.0],
        "longitude": [8.0, 8.1, 8.4, 9.7, 9.0, 9.5, 10.2, 9.7, 10.3, 9.7],
        "start_date": "2023-01-01",
        "end_date": "2025-11-10",
        "hourly": ["temperature_2m",
                   "cloud_cover",
                   "wind_speed_10m",
                   "wind_direction_10m"],
        "models": "era5",
    }

    API_URL = "https://archive-api.open-meteo.com/v1/archive"
    weather_data = OpenMeteoHandler(API_URL, **params)

    # Merge weather data with features_df
    features_df = features_df.merge(
        weather_data.data, on='datetime', how='left')

    # Print total and per-column NaN counts for weather_data
    if hasattr(weather_data, "data") and isinstance(weather_data.data, pd.DataFrame):
        nan_by_col = weather_data.data.isna().sum()
        total_nans = int(nan_by_col.sum())
        print(f"Total NaN values in weather_data: {total_nans}")
        print("NaN by column:")
        for col, cnt in nan_by_col.items():
            print(f"  {col}: {int(cnt)}")
    else:
        print("weather_data does not contain a pandas DataFrame in .data")

    ##########################################################################

    print("=" * 60)
    print("Final feature set overview")
    print("=" * 60)

    print(features_df.head())
    print("Shape features: ", features_df.shape)

    # Save features_df as parquet file relative to current file location
    output_path = (Path(__file__).resolve().parent.parent.parent /
                   "data/processed/imbalance_data.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path.with_suffix('.csv'), index=False)
    features_df.to_parquet(output_path, engine="pyarrow", index=False)
    print("Final shape features: ", features_df.shape)
