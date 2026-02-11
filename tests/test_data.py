import pandas as pd
from day_ahead_v2.data import DataHandler


def test_data_handler():
    """Test the DataHandler class."""
    data_handler = DataHandler()
    assert data_handler is not None
    assert hasattr(data_handler, "data")
    assert isinstance(data_handler.data, pd.DataFrame)
