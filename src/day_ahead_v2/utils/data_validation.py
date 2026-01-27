#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File name: data_validation.py
Author: Yannick Heiser
Created: 2025-11-27
Version: 1.0
Description:
    Data validation report structures.

Contact: yahei@dtu.dk
Dependencies:
    - dataclasses, typing, logging
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any
import logging
import json

logger = logging.getLogger(__name__)

@dataclass
class ShapeInfo:
    """Basic information about the DataFrame shape."""
    rows: int
    columns: int


@dataclass
class MissingColumnInfo:
    """Missing value stats for a single column."""
    count: int
    percentage: float


@dataclass
class MissingValuesInfo:
    """Missing values overview across the dataset."""
    total_missing: int
    by_column: Dict[str, MissingColumnInfo]


@dataclass
class DuplicateInfo:
    """Duplicate row statistics."""
    count: int
    percentage: float


@dataclass
class MemoryUsageInfo:
    """Memory usage statistics."""
    total_mb: float
    by_column_kb: Dict[str, float]


@dataclass
class NumericSummaryStats:
    """Summary statistics for a numeric column."""
    count: int
    mean: float
    std: float
    min: float
    q25: float
    q50: float
    q75: float
    max: float
    zeros: int
    negative: int

@dataclass
class ValidationReport:
    """Validation report for any loaded data."""
    data_type: str
    shape: ShapeInfo | None = None
    missing_values: MissingValuesInfo | None = None
    data_types: Dict[str, str] | None = None
    duplicates: DuplicateInfo | None = None
    memory_usage: MemoryUsageInfo | None = None
    numeric_summary: Dict[str, NumericSummaryStats] | None = None
    column_completeness: Dict[str, float] | None = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


    def summary(self) -> str:
        lines = [f"Data Type: {self.data_type}"]

        if self.shape:
            lines.append(f"Shape: {self.shape.rows} rows, {self.shape.columns} columns")

        if self.missing_values:
            total = self.missing_values.total_missing
            lines.append(f"Missing Values: {total}")
            for col, info in self.missing_values.by_column.items():
                lines.append(f"  {col}: {info.count} ({info.percentage:.2f}%)")
        else:
            lines.append("Missing Values: None")

        if self.duplicates:
            lines.append(f"Duplicate Rows: {self.duplicates.count} ({self.duplicates.percentage:.2f}%)")
        else:
            lines.append("Duplicate Rows: None")

        if self.memory_usage:
            lines.append(f"Memory Usage: {self.memory_usage.total_mb:.2f} MB")

        return "\n".join(lines)

    def log(self) -> None:
        for line in self.summary().split("\n"):
            logger.info(line)
