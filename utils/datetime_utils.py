"""
Datetime Utilities - Standardized Timezone Handling

This module provides consistent timezone handling across the Bull Machine codebase.
Eliminates manual timezone stripping and prevents timezone-related bugs.
"""

from datetime import datetime
from typing import Union
import pandas as pd


def to_timezone_naive(dt: Union[datetime, pd.Timestamp]) -> datetime:
    """
    Convert timezone-aware datetime to timezone-naive datetime.

    This is necessary for compatibility with macro data which uses naive datetimes.

    Args:
        dt: Datetime object (aware or naive)

    Returns:
        Timezone-naive datetime object

    Examples:
        >>> import pandas as pd
        >>> from datetime import datetime
        >>> dt_aware = pd.Timestamp('2024-01-01 12:00:00', tz='UTC')
        >>> dt_naive = to_timezone_naive(dt_aware)
        >>> dt_naive.tzinfo is None
        True
    """
    if dt is None:
        return None

    # Handle pandas Timestamp
    if isinstance(dt, pd.Timestamp):
        if dt.tzinfo is not None:
            return dt.tz_localize(None)
        return dt.to_pydatetime()

    # Handle datetime
    if hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        return dt.replace(tzinfo=None)

    return dt


def ensure_utc(dt: Union[datetime, pd.Timestamp]) -> pd.Timestamp:
    """
    Ensure datetime is timezone-aware in UTC.

    Args:
        dt: Datetime object (aware or naive)

    Returns:
        Timezone-aware pandas Timestamp in UTC

    Examples:
        >>> from datetime import datetime
        >>> dt_naive = datetime(2024, 1, 1, 12, 0, 0)
        >>> dt_utc = ensure_utc(dt_naive)
        >>> str(dt_utc.tzinfo)
        'UTC'
    """
    if dt is None:
        return None

    ts = pd.Timestamp(dt)

    if ts.tzinfo is None:
        return ts.tz_localize('UTC')

    return ts.tz_convert('UTC')


def align_timezone(dt1: datetime, dt2: datetime) -> tuple[datetime, datetime]:
    """
    Align two datetimes to same timezone state (both naive or both aware).

    Args:
        dt1: First datetime
        dt2: Second datetime

    Returns:
        Tuple of (dt1_aligned, dt2_aligned) with matching timezone state

    Examples:
        >>> from datetime import datetime
        >>> import pandas as pd
        >>> dt1 = datetime(2024, 1, 1, 12, 0, 0)
        >>> dt2 = pd.Timestamp('2024-01-01 12:00:00', tz='UTC')
        >>> aligned1, aligned2 = align_timezone(dt1, dt2)
        >>> aligned1.tzinfo is None and aligned2.tzinfo is None
        True
    """
    dt1_aware = hasattr(dt1, 'tzinfo') and dt1.tzinfo is not None
    dt2_aware = hasattr(dt2, 'tzinfo') and dt2.tzinfo is not None

    if dt1_aware and not dt2_aware:
        # Make both naive
        return to_timezone_naive(dt1), dt2
    elif not dt1_aware and dt2_aware:
        # Make both naive
        return dt1, to_timezone_naive(dt2)

    return dt1, dt2
