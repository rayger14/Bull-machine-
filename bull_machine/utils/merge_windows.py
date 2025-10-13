#!/usr/bin/env python3
"""
Window Merge Utility - Bull Machine v1.8.4

Merges overlapping candidate windows to prevent redundant bar processing.
Critical for batch mode performance - prevents overlap explosion.

Example:
    306 candidates with ±48 bar windows = 59,058 bar-visits
    After merge = ~2,500 bars (actual dataset size)
"""

from typing import List, Tuple


def merge_windows(
    intervals: List[Tuple[int, int]],
    min_gap: int = 0,
    expand: int = 0
) -> List[Tuple[int, int]]:
    """
    Merge overlapping [start, end] windows after optional symmetric expand.

    Args:
        intervals: List of (start, end) index pairs
        min_gap: Merge if gap between windows ≤ this value (default: 0)
        expand: Expand each window by this amount before merging (default: 0)

    Returns:
        List of merged (start, end) index pairs

    Example:
        >>> merge_windows([(100, 150), (140, 200), (500, 550)], min_gap=5)
        [(100, 200), (500, 550)]

        >>> merge_windows([(100, 110), (200, 210)], expand=50)
        [(50, 260)]  # Windows now overlap after expansion
    """
    if not intervals:
        return []

    # Expand windows symmetrically
    expanded = [(s - expand, e + expand) for (s, e) in intervals]

    # Sort by start index
    expanded.sort()

    # Merge overlapping intervals
    merged = []
    for s, e in expanded:
        if not merged or s > merged[-1][1] + min_gap:
            # No overlap - start new window
            merged.append([s, e])
        else:
            # Overlap - extend current window
            merged[-1][1] = max(merged[-1][1], e)

    # Clamp to non-negative indices
    return [(max(0, s), e) for s, e in merged]


def calculate_coverage(
    merged_windows: List[Tuple[int, int]],
    total_bars: int
) -> float:
    """
    Calculate what percentage of total bars are covered by merged windows.

    Args:
        merged_windows: List of merged (start, end) index pairs
        total_bars: Total number of bars in dataset

    Returns:
        Coverage ratio (0.0 to 1.0)

    Example:
        >>> calculate_coverage([(0, 500), (1000, 1500)], 2500)
        0.4  # 1000 bars covered out of 2500 total
    """
    if not merged_windows or total_bars <= 0:
        return 0.0

    covered_bars = sum(e - s + 1 for s, e in merged_windows)
    return min(1.0, covered_bars / total_bars)


def calculate_density(
    num_candidates: int,
    total_bars: int
) -> float:
    """
    Calculate candidate density (candidates per bar).

    Args:
        num_candidates: Number of candidate timestamps
        total_bars: Total number of bars in dataset

    Returns:
        Density ratio (0.0 to 1.0)

    Example:
        >>> calculate_density(300, 2500)
        0.12  # 12% of bars are candidates
    """
    if total_bars <= 0:
        return 0.0

    return min(1.0, num_candidates / total_bars)


def should_fallback_to_full(
    merged_windows: List[Tuple[int, int]],
    num_candidates: int,
    total_bars: int,
    coverage_threshold: float = 0.65,
    density_threshold: float = 0.15
) -> Tuple[bool, str]:
    """
    Determine if batch mode should fallback to full replay.

    Fallback when:
    - Coverage > 65% (batch mode processing too many bars anyway)
    - Density > 15% (too many candidates, not selective enough)

    Args:
        merged_windows: List of merged window intervals
        num_candidates: Number of candidates generated
        total_bars: Total bars in dataset
        coverage_threshold: Max coverage before fallback (default: 0.65)
        density_threshold: Max density before fallback (default: 0.15)

    Returns:
        (should_fallback, reason) tuple

    Example:
        >>> should_fallback_to_full([(0, 2000)], 400, 2500)
        (True, "coverage 0.80 > 0.65")
    """
    coverage = calculate_coverage(merged_windows, total_bars)
    density = calculate_density(num_candidates, total_bars)

    if coverage > coverage_threshold:
        return True, f"coverage {coverage:.2f} > {coverage_threshold}"

    if density > density_threshold:
        return True, f"density {density:.2f} > {density_threshold}"

    return False, ""


if __name__ == '__main__':
    # Quick validation tests
    print("Testing merge_windows utility...")

    # Test 1: No overlap
    intervals = [(100, 150), (300, 350), (500, 550)]
    merged = merge_windows(intervals)
    assert merged == [(100, 150), (300, 350), (500, 550)], "No overlap test failed"
    print("✅ Test 1: No overlap")

    # Test 2: Full overlap
    intervals = [(100, 150), (120, 180), (160, 200)]
    merged = merge_windows(intervals)
    assert merged == [(100, 200)], "Full overlap test failed"
    print("✅ Test 2: Full overlap")

    # Test 3: With expansion
    intervals = [(100, 110), (200, 210)]
    merged = merge_windows(intervals, expand=50)
    assert merged == [(50, 260)], "Expansion test failed"
    print("✅ Test 3: With expansion")

    # Test 4: Coverage calculation
    coverage = calculate_coverage([(0, 500), (1000, 1500)], 2500)
    assert abs(coverage - 0.4) < 0.01, "Coverage calculation failed"
    print("✅ Test 4: Coverage calculation")

    # Test 5: Density calculation
    density = calculate_density(300, 2500)
    assert abs(density - 0.12) < 0.01, "Density calculation failed"
    print("✅ Test 5: Density calculation")

    # Test 6: Fallback logic
    should_fb, reason = should_fallback_to_full([(0, 2000)], 400, 2500)
    assert should_fb and "coverage" in reason, "Fallback test failed"
    print("✅ Test 6: Fallback logic")

    print("\n✅ All tests passed!")
