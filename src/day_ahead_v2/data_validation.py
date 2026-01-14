from dataclasses import dataclass
from typing import Dict, Optional, Any


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
    """Top-level validation report for any loaded data."""
    data_type: str
    shape: Optional[ShapeInfo] = None
    missing_values: Optional[MissingValuesInfo] = None
    data_types: Optional[Dict[str, str]] = None
    duplicates: Optional[DuplicateInfo] = None
    memory_usage: Optional[MemoryUsageInfo] = None
    numeric_summary: Optional[Dict[str, NumericSummaryStats]] = None
    column_completeness: Optional[Dict[str, float]] = None
    message: Optional[str] = None