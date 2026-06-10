import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))
from pipeline import _align_to, _wmon_norm


def test_align_to_reindex():
    src_idx = pd.date_range("2023-01-09", periods=5, freq="W-MON")
    tgt_idx = pd.date_range("2023-01-02", periods=7, freq="W-MON")
    src = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=src_idx)
    result = _align_to(src, tgt_idx)
    assert len(result) == 7
    assert result.index.equals(tgt_idx)
    assert result.iloc[0] == 0.0   # fill_value=0 for missing prefix


def test_wmon_norm_datetime():
    idx = pd.date_range("2023-01-04", periods=3, freq="W-WED")
    result = _wmon_norm(idx)
    assert all(ts.weekday() == 0 for ts in result)


def test_wmon_norm_period():
    idx = pd.period_range("2023-01", periods=3, freq="W")
    result = _wmon_norm(idx)
    assert len(result) == 3
