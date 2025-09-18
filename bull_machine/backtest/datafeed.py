
from typing import Dict
import pandas as pd

TF_MAP = {
    '1m':'1T','5m':'5T','15m':'15T','30m':'30T',
    '1H':'1h','2H':'2h','4H':'4h','8H':'8h','12H':'12h',
    '1D':'1D','1W':'1W','1M':'1M'
}

class DataFeed:
    def __init__(self, sources: Dict[str,str], tz: str = 'UTC'):
        self.sources = sources
        self.frames: Dict[str, pd.DataFrame] = {}
        self.tz = tz
        for sym, path in sources.items():
            df = pd.read_csv(path)
            df.columns = [c.lower().strip() for c in df.columns]
            if 'timestamp' in df.columns:
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                except Exception:
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                df = df.set_index('timestamp')
            else:
                df.index = pd.date_range('2018-01-01', periods=len(df), freq='1h')
            for col in ['open','high','low','close']:
                if col not in df.columns: raise ValueError(f"{sym}: missing column '{col}'")
            if 'volume' not in df.columns:
                df['volume'] = 0.0
            df = df[['open','high','low','close','volume']].sort_index()
            self.frames[sym] = df.tz_localize(self.tz, nonexistent='shift_forward', ambiguous='NaT')

    def resample(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        freq = TF_MAP.get(tf, '1H')
        ohlc = df.resample(freq, label='right', closed='right').agg({
            'open':'first','high':'max','low':'min','close':'last','volume':'sum'
        }).dropna()
        return ohlc

    def window(self, symbol: str, tf: str, end_idx: int, lookback: int = 250) -> pd.DataFrame:
        base = self.frames[symbol]
        df_tf = self.resample(base, tf)
        if end_idx <= 0: end_idx = len(df_tf)
        start = max(0, end_idx - lookback)
        return df_tf.iloc[start:end_idx]
