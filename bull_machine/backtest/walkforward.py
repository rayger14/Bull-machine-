
from typing import Iterator, Tuple

def rolling_windows(n_bars: int, train: int, test: int, step: int) -> Iterator[Tuple[int,int,int,int]]:
    start = 0
    while start + train + test <= n_bars:
        yield (start, start+train, start+train, start+train+test)
        start += step
