from dataclasses import dataclass


# Placeholder adapter. Claude should call your v1.3 pipeline using a DataFrame/Series entrypoint.
@dataclass
class Bar:
    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Series:
    bars: list[Bar]
    timeframe: str
    symbol: str


def df_to_series(symbol: str, tf: str, df) -> Series:
    bars = [
        Bar(
            ts=int(i.value // 10**9),
            open=float(r.open),
            high=float(r.high),
            low=float(r.low),
            close=float(r.close),
            volume=float(getattr(r, "volume", 0.0)),
        )
        for i, r in df.iterrows()
    ]
    return Series(bars=bars, timeframe=tf, symbol=symbol)


def strategy_from_df(symbol: str, tf: str, df_window, balance: float, config_path: str):
    # TODO: Implement: call real v1.3 pipeline and map to {'action': 'long|short|exit|flat', 'size': float}
    return {"action": "flat"}
