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
Dependencies: dataclasses, pandas, os, typing
"""

from dataclasses import dataclass, field
import os
import pandas as pd
from typing import Dict, Any, Union, Optional


@dataclass
class DataImporter:
    """
    A class to import data from various file formats into a pandas DataFrame.
    Supports CSV, Excel, JSON, and Parquet files.
    If the file format is not recognized, it reads the file as plain text.

    Attributes:
        filename (str): The name of the file to import.
        arguments (Dict[str, Any]): Additional arguments to pass to the
            pandas read function.
    """
    filename: str
    arguments: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self._data = self._load_data()

    @property
    def data(self) -> Union[pd.DataFrame, str]:
        """
        Access the loaded data.
        
        Returns:
            Union[pd.DataFrame, str]: The loaded data as a pandas DataFrame
            for supported formats, or as a string for unsupported formats.
        """
        return self._data

    def _load_data(self) -> Union[pd.DataFrame, str]:
        """
        Load data from the specified file into a pandas DataFrame.

        Raises:
            FileNotFoundError: If the file is not found.

        Returns:
            Union[pd.DataFrame, str]: The loaded data as a DataFrame or
            plain text.
        """
        data_path = os.path.join(
            os.path.dirname(__file__), '../data', self.filename
        )
        
        # Check if file exists
        if not os.path.isfile(data_path):
            dir_path = os.path.dirname(data_path)
            raise FileNotFoundError(
                f"File '{self.filename}' not found in '{dir_path}'"
            )
        
        # Determine file extension and use appropriate pandas reader
        file_extension = os.path.splitext(self.filename)[1].lower()
        if file_extension == '.csv':
            return pd.read_csv(data_path, **self.arguments)
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(data_path, **self.arguments)
        elif file_extension == '.json':
            return pd.read_json(data_path, **self.arguments)
        elif file_extension == '.parquet':
            return pd.read_parquet(data_path, **self.arguments)
        else:
            # For other file types, fall back to reading as text
            with open(data_path, 'r') as f:
                return f.read()
            
    def validate_data(self) -> Dict[str, Any]:
        """
        Validate the loaded data and return a comprehensive data quality
        report.
        
        Returns:
            Dict[str, Any]: A dictionary containing data validation metrics
            including:
                - data_type: Type of the loaded data (DataFrame or str)
                - shape: Dimensions of the data (if DataFrame)
                - missing_values: Count and percentage of missing values
                  per column
                - data_types: Data types of each column
                - duplicates: Number of duplicate rows
                - memory_usage: Memory usage information
                - numeric_summary: Summary statistics for numeric columns
                - column_completeness: Percentage of non-missing values
                  per column
                - text_info: Length and line count (if text data)
        """
        validation_report: Dict[str, Any] = {}
        
        # Check if data is a DataFrame or text
        validation_report['data_type'] = type(self._data).__name__
        
        if isinstance(self._data, pd.DataFrame):
            # Basic shape information
            validation_report['shape'] = {
                'rows': len(self._data),
                'columns': len(self._data.columns)
            }
            
            # Missing values analysis
            missing_counts = self._data.isnull().sum()
            missing_percentages = (missing_counts / len(self._data)) * 100
            
            validation_report['missing_values'] = {
                'total_missing': missing_counts.sum(),
                'by_column': {
                    col: {
                        'count': int(missing_counts[col]),
                        'percentage': round(missing_percentages[col], 2)
                    }
                    for col in self._data.columns
                    if missing_counts[col] > 0
                }
            }
            
            # Data types
            validation_report['data_types'] = {
                col: str(dtype) for col, dtype in self._data.dtypes.items()
            }
            
            # Duplicate rows
            dup_sum = self._data.duplicated().sum()
            validation_report['duplicates'] = {
                'count': dup_sum,
                'percentage': round((dup_sum / len(self._data)) * 100, 2)
            }
            
            # Memory usage
            memory_usage = self._data.memory_usage(deep=True)
            validation_report['memory_usage'] = {
                'total_mb': round(memory_usage.sum() / 1024 / 1024, 2),
                'by_column_kb': {
                    col: round(memory_usage[col] / 1024, 2)
                    for col in memory_usage.index
                }
            }
            
            # Summary statistics for numeric columns
            numeric_cols = self._data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                validation_report['numeric_summary'] = {
                    col: {
                        'count': int(self._data[col].count()),
                        'mean': round(self._data[col].mean(), 2),
                        'std': round(self._data[col].std(), 2),
                        'min': self._data[col].min(),
                        '25%': self._data[col].quantile(0.25),
                        '50%': self._data[col].quantile(0.5),
                        '75%': self._data[col].quantile(0.75),
                        'max': self._data[col].max(),
                        'zeros': int((self._data[col] == 0).sum()),
                        'negative': int((self._data[col] < 0).sum())
                    }
                    for col in numeric_cols
                }
            
            # Column completeness
            validation_report['column_completeness'] = {
                col: round((1 - missing_percentages[col] / 100) * 100, 2)
                for col in self._data.columns
            }
            
        else:
            # For text data
            validation_report['text_info'] = {
                'length': len(str(self._data)),
                'lines': str(self._data).count('\n') + 1 if self._data else 0
            }
        
        return validation_report
    
    def print_validation_summary(self) -> None:
        """
        Print a formatted summary of the data validation report.
        """
        report = self.validate_data()
        
        print("=" * 60)
        print(f"DATA VALIDATION REPORT FOR: {self.filename}")
        print("=" * 60)
        
        if report['data_type'] == 'DataFrame':
            shape = report['shape']
            print(f"📊 Data Shape: {shape['rows']} rows × "
                  f"{shape['columns']} columns")
            print(f"💾 Memory Usage: {report['memory_usage']['total_mb']} MB")
            
            # Missing values summary
            total_missing = report['missing_values']['total_missing']
            if total_missing > 0:
                print(f"❌ Missing Values: {total_missing} total")
                print("   Columns with missing values:")
                for col, info in report['missing_values']['by_column'].items():
                    print(f"   • {col}: {info['count']} "
                          f"({info['percentage']}%)")
            else:
                print("✅ Missing Values: None")
            
            # Duplicates
            dup_count = report['duplicates']['count']
            if dup_count > 0:
                dup_pct = report['duplicates']['percentage']
                print(f"🔄 Duplicate Rows: {dup_count} ({dup_pct}%)")
            else:
                print("✅ Duplicate Rows: None")
            
            # Data types
            unique_types = len(set(report['data_types'].values()))
            print(f"🏷️  Data Types: {unique_types} unique types")
            
            # Numeric summary
            if 'numeric_summary' in report:
                print(f"🔢 Numeric Columns: {len(report['numeric_summary'])}")
                for col, stats in report['numeric_summary'].items():
                    if stats['negative'] > 0 or stats['zeros'] > 0:
                        print(f"   • {col}: {stats['zeros']} zeros, "
                              f"{stats['negative']} negative values")
        
        else:
            length = report['text_info']['length']
            lines = report['text_info']['lines']
            print(f"📄 Text Data: {length} characters, {lines} lines")
        
        print("=" * 60)

    def preview(self, n_rows: int = 5, show_info: bool = True) -> None:
        """
        Display a quick preview of the loaded data.
        
        Args:
            n_rows (int): Number of rows to display from head and tail.
                         Default is 5.
            show_info (bool): Whether to show data info summary.
                             Default is True.
        """
        print("=" * 70)
        print(f"📋 DATA PREVIEW: {self.filename}")
        print("=" * 70)
        
        if isinstance(self._data, pd.DataFrame):
            if show_info:
                # Basic info
                print(f"📊 Shape: {self._data.shape[0]} rows × "
                      f"{self._data.shape[1]} columns")
                memory_mb = round(
                    self._data.memory_usage(deep=True).sum() / 1024 / 1024, 2
                )
                print(f"💾 Memory: {memory_mb} MB")
                
                # Column info
                print(f"🏷️  Columns: {list(self._data.columns)}")
                print(f"🔢 Data Types: {dict(self._data.dtypes)}")
                print()
            
            # Show head
            print(f"📤 First {min(n_rows, len(self._data))} rows:")
            print("-" * 50)
            print(self._data.head(n_rows))
            print()
            
            # Show tail if data has more than n_rows
            if len(self._data) > n_rows:
                print(f"📥 Last {min(n_rows, len(self._data))} rows:")
                print("-" * 50)
                print(self._data.tail(n_rows))
                print()
            
            # Basic statistics for numeric columns
            numeric_cols = self._data.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0 and show_info:
                print("📈 Numeric Summary:")
                print("-" * 50)
                print(self._data[numeric_cols].describe())
                print()
            
            # Missing values summary
            if show_info:
                missing = self._data.isnull().sum()
                if missing.any():
                    print("❌ Missing Values:")
                    print("-" * 50)
                    missing_data = missing[missing > 0]
                    missing_summary = missing_data.sort_values(ascending=False)
                    for col, count in missing_summary.items():
                        pct = round((count / len(self._data)) * 100, 2)
                        print(f"   {col}: {count} ({pct}%)")
                    print()
                else:
                    print("✅ No missing values found")
                    print()
        
        else:
            # For text data
            print("📄 Text Data Preview:")
            print("-" * 50)
            text_preview = str(self._data)
            lines = text_preview.split('\n')
            
            # Show first few lines
            preview_lines = min(n_rows * 2, len(lines))
            for i, line in enumerate(lines[:preview_lines]):
                print(f"{i+1:3d}: {line}")
            
            if len(lines) > preview_lines:
                print(f"... ({len(lines) - preview_lines} more lines)")

            print()
            print(f"📊 Total: {len(text_preview)} characters, "
                  f"{len(lines)} lines")

        print("=" * 70)

    def head(self, n: int = 5) -> Union[pd.DataFrame, str]:
        """
        Return the first n rows of the data.

        Args:
            n (int): Number of rows to return. Default is 5.
 
        Returns:
            Union[pd.DataFrame, str]: First n rows if DataFrame,
            or first n lines if text.
        """
        if isinstance(self._data, pd.DataFrame):
            return self._data.head(n)
        else:
            lines = str(self._data).split('\n')
            return '\n'.join(lines[:n])

    def tail(self, n: int = 5) -> Union[pd.DataFrame, str]:
        """
        Return the last n rows of the data.
        
        Args:
            n (int): Number of rows to return. Default is 5.
            
        Returns:
            Union[pd.DataFrame, str]: Last n rows if DataFrame,
            or last n lines if text.
        """
        if isinstance(self._data, pd.DataFrame):
            return self._data.tail(n)
        else:
            lines = str(self._data).split('\n')
            return '\n'.join(lines[-n:])

    def info(self) -> None:
        """
        Display concise summary information about the data.
        """
        if isinstance(self._data, pd.DataFrame):
            print(f"📋 Data Info for: {self.filename}")
            print("-" * 40)
            self._data.info()
        else:
            text_data = str(self._data)
            lines = text_data.split('\n')
            print(f"📄 Text Info for: {self.filename}")
            print("-" * 40)
            print(f"Length: {len(text_data)} characters")
            print(f"Lines: {len(lines)}")
            print(f"Type: {type(self._data).__name__}")

    @property
    def shape(self) -> Union[tuple, str]:
        """
        Get the shape of the data.
        
        Returns:
            Union[tuple, str]: Shape tuple for DataFrame,
            or description for text data.
        """
        if isinstance(self._data, pd.DataFrame):
            return self._data.shape
        else:
            lines = str(self._data).split('\n')
            return f"{len(str(self._data))} characters, {len(lines)} lines"

    def transform_data(self, **kwargs) -> 'DataImporter':
        """
        Apply data transformations and return a new instance.
        
        Args:
            **kwargs: Transformation options including:
                - drop_duplicates (bool): Remove duplicate rows
                - drop_missing_threshold (float): Drop columns with missing
                  values above this threshold (0.0-1.0)
                - fill_missing (str/dict): Fill missing values
                  ('mean', 'median', 'mode', 'forward', 'backward', 'zero', 'interpolate')
                - convert_types (dict): Convert column types
                  {column: target_type}
                - normalize_columns (list): Columns to normalize (0-1 scale)
                - standardize_columns (list): Columns to standardize
                  (z-score)
                - drop_columns (list): Columns to drop
                - rename_columns (dict): Rename columns {old: new}
                
        Returns:
            DataImporter: New instance with transformed data.
        """
        if not isinstance(self._data, pd.DataFrame):
            print("⚠️  Data transformation only available for DataFrame data")
            return self
            
        # Create a copy to avoid modifying original data
        df = self._data.copy()
        log = []
        
        # Drop columns
        if 'drop_columns' in kwargs:
            cols = kwargs['drop_columns']
            df = df.drop(columns=cols, errors='ignore')
            log.append(f"Dropped columns: {cols}")
        
        # Rename columns
        if 'rename_columns' in kwargs:
            rename_dict = kwargs['rename_columns']
            df = df.rename(columns=rename_dict)
            log.append(f"Renamed columns: {rename_dict}")
        
        # Drop duplicates
        if kwargs.get('drop_duplicates', False):
            original_rows = len(df)
            df = df.drop_duplicates()
            removed = original_rows - len(df)
            log.append(f"Removed {removed} duplicate rows")
        
        # Fill missing values
        if 'fill_missing' in kwargs:
            method = kwargs['fill_missing']
            if method == 'mean':
                numeric_cols = df.select_dtypes(include=['number']).columns
                df[numeric_cols] = df[numeric_cols].fillna(
                    df[numeric_cols].mean()
                )
            elif method == 'median':
                numeric_cols = df.select_dtypes(include=['number']).columns
                df[numeric_cols] = df[numeric_cols].fillna(
                    df[numeric_cols].median()
                )
            elif method == 'mode':
                for col in df.columns:
                    mode_val = df[col].mode()
                    if not mode_val.empty:
                        df[col] = df[col].fillna(mode_val[0])
            elif method == 'forward':
                df = df.fillna(method='ffill')
            elif method == 'backward':
                df = df.fillna(method='bfill')
            elif method == 'zero':
                df = df.fillna(0)
            elif method == 'interpolate':
                df = df.interpolate()
            log.append(f"Filled missing values using: {method}")
        
        # Convert types
        if 'convert_types' in kwargs:
            type_dict = kwargs['convert_types']
            for col, dtype in type_dict.items():
                if col in df.columns:
                    try:
                        df[col] = df[col].astype(dtype)
                        log.append(f"Converted {col} to {dtype}")
                    except Exception as e:
                        log.append(f"Failed to convert {col}: {str(e)[:30]}")
        
        # Normalize columns
        if 'normalize_columns' in kwargs:
            cols = kwargs['normalize_columns']
            for col in cols:
                if col in df.columns and df[col].dtype in ['int64', 'float64']:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val != min_val:
                        df[col] = (df[col] - min_val) / (max_val - min_val)
                        log.append(f"Normalized {col}")
        
        # Create new instance
        new_instance = DataImporter.__new__(DataImporter)
        new_instance.filename = f"transformed_{self.filename}"
        new_instance.arguments = self.arguments.copy()
        new_instance._data = df
        
        # Print summary
        print("🔄 TRANSFORMATION SUMMARY:")
        print("-" * 40)
        for entry in log:
            print(f"✅ {entry}")
        print(f"📊 Final shape: {df.shape}")
        
        return new_instance

    def save(self, filename: Union[str, None] = None,
             format: str = 'csv',
             directory: Union[str, None] = None, **kwargs) -> str:
        """
        Save the data to a file in the specified format.
        
        Args:
            filename (str, optional): Name for the saved file. If None,
                uses original filename with new format.
            format (str): Output format ('csv', 'parquet', 'excel', 'json').
                Default is 'csv'.
            directory (str, optional): Directory to save the file. If None,
                saves to the same directory as original data.
            **kwargs: Additional arguments to pass to pandas save function.
                
        Returns:
            str: Path to the saved file.
            
        Raises:
            ValueError: If trying to save non-DataFrame data in
                DataFrame formats.
            OSError: If the directory doesn't exist or can't be created.
        """
        if not isinstance(self._data, pd.DataFrame):
            if format in ['csv', 'parquet', 'excel']:
                raise ValueError(
                    f"Cannot save text data as {format}. "
                    "DataFrame required for this format."
                )
        
        # Determine output directory
        if directory is None:
            directory = os.path.join(os.path.dirname(__file__), '../data')
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            base_name = os.path.splitext(self.filename)[0]
            filename = f"{base_name}_saved.{format}"
        elif not filename.endswith(f'.{format}'):
            filename = f"{filename}.{format}"
        
        # Full path for the output file
        output_path = os.path.join(directory, filename)
        
        try:
            if isinstance(self._data, pd.DataFrame):
                if format == 'csv':
                    # Default CSV settings
                    csv_kwargs = {'index': False}
                    csv_kwargs.update(kwargs)
                    self._data.to_csv(output_path, **csv_kwargs)
                    
                elif format == 'parquet':
                    # Default Parquet settings
                    parquet_kwargs = {'index': False}
                    parquet_kwargs.update(kwargs)
                    self._data.to_parquet(output_path, **parquet_kwargs)
                    
                elif format in ['excel', 'xlsx']:
                    # Default Excel settings
                    excel_kwargs = {'index': False}
                    excel_kwargs.update(kwargs)
                    if not output_path.endswith('.xlsx'):
                        output_path = output_path.replace('.excel', '.xlsx')
                    self._data.to_excel(output_path, **excel_kwargs)
                    
                elif format == 'json':
                    # Default JSON settings
                    json_kwargs = {'orient': 'records', 'indent': 2}
                    json_kwargs.update(kwargs)
                    self._data.to_json(output_path, **json_kwargs)
                    
                else:
                    raise ValueError(
                        f"Unsupported format: {format}. "
                        "Supported formats: csv, parquet, excel, json"
                    )
            else:
                # Save text data
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(str(self._data))
            
            # Print success message
            if isinstance(self._data, pd.DataFrame):
                rows, cols = self._data.shape
                print(f"💾 Successfully saved {rows} rows × {cols} columns")
            else:
                char_count = len(str(self._data))
                print(f"💾 Successfully saved {char_count} characters")
            
            print(f"📁 File saved as: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Error saving file: {str(e)}")
            raise