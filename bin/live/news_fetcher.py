#!/usr/bin/env python3
"""
News Fetcher for Bull Machine Trading System

Fetches crypto news headlines and computes sentiment scores for use in
the live trading pipeline. Primary source is CryptoPanic API (free tier);
falls back to CoinDesk and CoinTelegraph RSS feeds when no API key is
available or the API is unreachable.

Results are cached in memory with a 1-hour TTL to respect rate limits
(CryptoPanic free tier: 200 requests/day).

Usage:
    from bin.live.news_fetcher import NewsFetcher

    fetcher = NewsFetcher()
    latest = fetcher.get_latest(count=10)
    sentiment = fetcher.get_aggregate_sentiment()

    # Standalone test:
    python3 bin/live/news_fetcher.py

Author: Claude Code (System Architect)
Date: 2026-03-03
"""

import json
import logging
import os
import ssl
import threading
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Dict, List, Optional
from urllib.error import HTTPError, URLError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional feedparser for RSS (graceful fallback to stdlib XML parsing)
# ---------------------------------------------------------------------------
FEEDPARSER_AVAILABLE = False
try:
    import feedparser
    FEEDPARSER_AVAILABLE = True
except ImportError:
    logger.debug("feedparser not installed; using stdlib XML fallback for RSS.")

# ---------------------------------------------------------------------------
# Keyword sentiment lists
# ---------------------------------------------------------------------------
BULLISH_KEYWORDS = [
    "rally", "surge", "soar", "bull", "bullish", "breakout", "all-time high",
    "ath", "pump", "moon", "gain", "record high", "upside", "buy", "buying",
    "accumulate", "accumulation", "adoption", "etf approved", "etf approval",
    "institutional", "inflow", "inflows", "upgrade", "optimistic", "positive",
    "recovery", "rebound", "support", "golden cross", "halving",
]

BEARISH_KEYWORDS = [
    "crash", "dump", "plunge", "bear", "bearish", "sell-off", "selloff",
    "sell off", "decline", "drop", "fall", "correction", "liquidation",
    "liquidated", "hack", "hacked", "exploit", "ban", "banned", "crackdown",
    "regulation", "sec", "lawsuit", "fraud", "scam", "rug pull", "rugpull",
    "outflow", "outflows", "downgrade", "pessimistic", "negative",
    "death cross", "capitulation", "fear", "panic", "warning",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CACHE_TTL_SECONDS = 3600  # 1 hour
MAX_CACHE_ITEMS = 50

CRYPTOPANIC_BASE_URL = "https://cryptopanic.com/api/free/v1/posts/"
COINDESK_RSS_URL = "https://www.coindesk.com/arc/outboundfeeds/rss/"
COINTELEGRAPH_RSS_URL = "https://cointelegraph.com/rss"

HTTP_TIMEOUT_SECONDS = 15
USER_AGENT = "BullMachine/1.0 NewsFetcher"


class NewsFetcher:
    """Fetches crypto news and computes sentiment scores.

    Thread-safe with internal locking on cache access. Automatically
    falls back from CryptoPanic API to RSS feeds on failure.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the NewsFetcher.

        Args:
            api_key: CryptoPanic API key. If None, reads from
                     CRYPTOPANIC_API_KEY environment variable.
        """
        self._api_key = api_key or os.environ.get("CRYPTOPANIC_API_KEY")
        self._cache: List[Dict] = []
        self._cache_timestamp: float = 0.0
        self._lock = threading.Lock()
        self._request_count = 0
        self._request_day: Optional[str] = None

        if self._api_key:
            logger.info("NewsFetcher initialized with CryptoPanic API key.")
        else:
            logger.info(
                "NewsFetcher initialized without API key; "
                "will use RSS fallback."
            )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_latest(self, count: int = 10) -> List[Dict]:
        """Return the latest news items with sentiment scores.

        Args:
            count: Maximum number of items to return.

        Returns:
            List of dicts, each containing:
                headline (str), source (str), published_at (str, ISO 8601),
                sentiment (str: 'bullish'|'bearish'|'neutral'),
                sentiment_score (float: -1.0 to 1.0), url (str).
        """
        items = self._get_cached_or_fetch()
        return items[:count]

    def get_aggregate_sentiment(self) -> Dict:
        """Return aggregate sentiment summary over cached headlines.

        Returns:
            Dict with bullish_count, bearish_count, neutral_count,
            avg_score (float), and summary (str).
        """
        items = self._get_cached_or_fetch()

        bullish_count = sum(1 for i in items if i["sentiment"] == "bullish")
        bearish_count = sum(1 for i in items if i["sentiment"] == "bearish")
        neutral_count = sum(1 for i in items if i["sentiment"] == "neutral")

        total = len(items)
        if total > 0:
            avg_score = sum(i["sentiment_score"] for i in items) / total
        else:
            avg_score = 0.0

        if total > 0:
            summary = f"{bullish_count}/{total} bullish"
        else:
            summary = "no data"

        return {
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "avg_score": round(avg_score, 3),
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _get_cached_or_fetch(self) -> List[Dict]:
        """Return cached items if fresh, otherwise fetch new data."""
        with self._lock:
            now = time.time()
            if self._cache and (now - self._cache_timestamp) < CACHE_TTL_SECONDS:
                return list(self._cache)

        # Fetch outside the lock to avoid blocking other threads
        items = self._fetch_all()

        with self._lock:
            self._cache = items[:MAX_CACHE_ITEMS]
            self._cache_timestamp = time.time()
            return list(self._cache)

    # ------------------------------------------------------------------
    # Fetch orchestration
    # ------------------------------------------------------------------

    def _fetch_all(self) -> List[Dict]:
        """Try CryptoPanic first, fall back to RSS on failure."""
        # Try CryptoPanic API if we have a key and haven't hit rate limit
        if self._api_key and self._check_rate_limit():
            try:
                items = self._fetch_cryptopanic()
                if items:
                    logger.info(
                        "Fetched %d items from CryptoPanic API.", len(items)
                    )
                    return items
            except Exception as exc:
                logger.warning("CryptoPanic API failed: %s", exc)

        # Fallback to RSS
        return self._fetch_rss_combined()

    def _check_rate_limit(self) -> bool:
        """Check whether we are within the daily rate limit (200/day)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            if self._request_day != today:
                self._request_day = today
                self._request_count = 0
            if self._request_count >= 200:
                logger.warning(
                    "CryptoPanic daily rate limit reached (%d requests).",
                    self._request_count,
                )
                return False
            return True

    def _increment_request_count(self) -> None:
        """Increment the daily request counter (thread-safe)."""
        with self._lock:
            self._request_count += 1

    # ------------------------------------------------------------------
    # CryptoPanic API
    # ------------------------------------------------------------------

    def _fetch_cryptopanic(self) -> List[Dict]:
        """Fetch headlines from CryptoPanic API."""
        url = (
            f"{CRYPTOPANIC_BASE_URL}"
            f"?currencies=BTC&filter=important"
            f"&auth_token={self._api_key}"
        )

        data = self._http_get_json(url)
        self._increment_request_count()

        results = data.get("results", [])
        items: List[Dict] = []

        for post in results:
            title = post.get("title", "")
            published = post.get("published_at", "")
            source_info = post.get("source", {})
            source_name = (
                source_info.get("title", "CryptoPanic")
                if isinstance(source_info, dict)
                else str(source_info)
            )
            post_url = post.get("url", "")

            # Compute sentiment from votes if available
            votes = post.get("votes", {})
            sentiment_score = self._score_from_votes(votes)

            # If no votes, fall back to keyword analysis
            if sentiment_score == 0.0 and not votes:
                sentiment_score = self._keyword_sentiment_score(title)

            sentiment_label = self._label_from_score(sentiment_score)

            items.append({
                "headline": title,
                "source": source_name,
                "published_at": self._normalize_timestamp(published),
                "sentiment": sentiment_label,
                "sentiment_score": round(sentiment_score, 3),
                "url": post_url,
            })

        return items

    @staticmethod
    def _score_from_votes(votes: Dict) -> float:
        """Compute a -1 to 1 sentiment score from CryptoPanic vote counts.

        Votes dict may contain: positive, negative, important, liked,
        disliked, lol, toxic, saved, comments.

        We use positive (bullish proxy) vs negative (bearish proxy).
        """
        if not votes or not isinstance(votes, dict):
            return 0.0

        positive = int(votes.get("positive", 0) or 0)
        negative = int(votes.get("negative", 0) or 0)

        total = positive + negative
        if total == 0:
            return 0.0

        # Net score: (pos - neg) / total => range [-1, 1]
        return (positive - negative) / total

    # ------------------------------------------------------------------
    # RSS feeds
    # ------------------------------------------------------------------

    def _fetch_rss_combined(self) -> List[Dict]:
        """Fetch and merge RSS items from CoinDesk and CoinTelegraph."""
        items: List[Dict] = []

        for feed_url, source_name in [
            (COINDESK_RSS_URL, "CoinDesk"),
            (COINTELEGRAPH_RSS_URL, "CoinTelegraph"),
        ]:
            try:
                feed_items = self._fetch_rss_feed(feed_url, source_name)
                items.extend(feed_items)
                logger.info(
                    "Fetched %d items from %s RSS.", len(feed_items), source_name
                )
            except Exception as exc:
                logger.warning("RSS fetch from %s failed: %s", source_name, exc)

        # Sort by published_at descending (most recent first)
        items.sort(key=lambda x: x.get("published_at", ""), reverse=True)
        return items[:MAX_CACHE_ITEMS]

    def _fetch_rss_feed(self, url: str, source_name: str) -> List[Dict]:
        """Parse a single RSS feed and return news items."""
        if FEEDPARSER_AVAILABLE:
            return self._parse_with_feedparser(url, source_name)
        return self._parse_with_stdlib(url, source_name)

    def _parse_with_feedparser(self, url: str, source_name: str) -> List[Dict]:
        """Parse RSS using the feedparser library."""
        raw_xml = self._http_get_text(url)
        feed = feedparser.parse(raw_xml)

        items: List[Dict] = []
        for entry in feed.entries[:30]:
            title = getattr(entry, "title", "")
            link = getattr(entry, "link", "")

            # feedparser normalizes dates into time_struct or string
            published = ""
            if hasattr(entry, "published"):
                published = entry.published
            elif hasattr(entry, "updated"):
                published = entry.updated

            sentiment_score = self._keyword_sentiment_score(title)
            sentiment_label = self._label_from_score(sentiment_score)

            items.append({
                "headline": title,
                "source": source_name,
                "published_at": self._normalize_timestamp(published),
                "sentiment": sentiment_label,
                "sentiment_score": round(sentiment_score, 3),
                "url": link,
            })

        return items

    def _parse_with_stdlib(self, url: str, source_name: str) -> List[Dict]:
        """Parse RSS using stdlib xml.etree.ElementTree (no feedparser)."""
        raw_xml = self._http_get_text(url)
        items: List[Dict] = []

        try:
            root = ET.fromstring(raw_xml)
        except ET.ParseError as exc:
            logger.warning("XML parse error for %s: %s", source_name, exc)
            return items

        # Standard RSS 2.0: <rss><channel><item>
        # Atom: <feed><entry>
        # Handle both with namespace-unaware search
        rss_items = root.findall(".//item")
        if not rss_items:
            # Try Atom format
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            rss_items = root.findall(".//atom:entry", ns)
            if not rss_items:
                # Try without namespace
                rss_items = root.findall(".//{http://www.w3.org/2005/Atom}entry")

        for item_elem in rss_items[:30]:
            title = self._xml_text(item_elem, "title")
            link = self._xml_text(item_elem, "link")
            pub_date = (
                self._xml_text(item_elem, "pubDate")
                or self._xml_text(item_elem, "published")
                or self._xml_text(item_elem, "updated")
            )

            # For Atom feeds, link may be in href attribute
            if not link:
                link_elem = item_elem.find("link")
                if link_elem is None:
                    link_elem = item_elem.find(
                        "{http://www.w3.org/2005/Atom}link"
                    )
                if link_elem is not None:
                    link = link_elem.get("href", "")

            sentiment_score = self._keyword_sentiment_score(title)
            sentiment_label = self._label_from_score(sentiment_score)

            items.append({
                "headline": title,
                "source": source_name,
                "published_at": self._normalize_timestamp(pub_date),
                "sentiment": sentiment_label,
                "sentiment_score": round(sentiment_score, 3),
                "url": link,
            })

        return items

    @staticmethod
    def _xml_text(element, tag: str) -> str:
        """Extract text from an XML child element, trying with/without namespaces."""
        child = element.find(tag)
        if child is None:
            child = element.find(f"{{http://www.w3.org/2005/Atom}}{tag}")
        if child is not None and child.text:
            return child.text.strip()
        return ""

    # ------------------------------------------------------------------
    # Keyword sentiment analysis
    # ------------------------------------------------------------------

    @staticmethod
    def _keyword_sentiment_score(headline: str) -> float:
        """Score a headline based on bullish/bearish keyword matches.

        Returns a float in [-1, 1]. Multiple keyword matches increase
        the magnitude, but the result is clamped.
        """
        if not headline:
            return 0.0

        lower = headline.lower()
        bullish_hits = sum(1 for kw in BULLISH_KEYWORDS if kw in lower)
        bearish_hits = sum(1 for kw in BEARISH_KEYWORDS if kw in lower)

        total_hits = bullish_hits + bearish_hits
        if total_hits == 0:
            return 0.0

        # Net score, each hit contributes 0.3, clamped to [-1, 1]
        raw_score = (bullish_hits - bearish_hits) * 0.3
        return max(-1.0, min(1.0, raw_score))

    @staticmethod
    def _label_from_score(score: float) -> str:
        """Convert a numeric sentiment score to a label."""
        if score > 0.1:
            return "bullish"
        elif score < -0.1:
            return "bearish"
        return "neutral"

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _http_get_json(self, url: str) -> Dict:
        """Fetch a URL and parse JSON response."""
        text = self._http_get_text(url)
        return json.loads(text)

    @staticmethod
    def _http_get_text(url: str, _max_redirects: int = 5) -> str:
        """Fetch a URL and return the response body as text.

        Follows 301/302/307/308 redirects up to _max_redirects times.
        """
        ctx = ssl.create_default_context()
        current_url = url

        for _ in range(_max_redirects):
            req = urllib.request.Request(
                current_url,
                headers={"User-Agent": USER_AGENT},
            )

            try:
                with urllib.request.urlopen(
                    req, timeout=HTTP_TIMEOUT_SECONDS, context=ctx
                ) as resp:
                    raw = resp.read()
                    encoding = resp.headers.get_content_charset() or "utf-8"
                    return raw.decode(encoding, errors="replace")
            except HTTPError as exc:
                if exc.code in (301, 302, 307, 308):
                    redirect_url = exc.headers.get("Location")
                    if redirect_url:
                        # Resolve relative redirect URLs
                        if redirect_url.startswith("/"):
                            from urllib.parse import urlparse
                            parsed = urlparse(current_url)
                            redirect_url = (
                                f"{parsed.scheme}://{parsed.netloc}"
                                f"{redirect_url}"
                            )
                        logger.debug(
                            "Following %d redirect: %s -> %s",
                            exc.code, current_url, redirect_url,
                        )
                        current_url = redirect_url
                        continue
                raise
            except ssl.SSLError:
                logger.debug(
                    "SSL verification failed for %s; retrying unverified.",
                    current_url,
                )
                ctx_noverify = ssl._create_unverified_context()
                req2 = urllib.request.Request(
                    current_url,
                    headers={"User-Agent": USER_AGENT},
                )
                with urllib.request.urlopen(
                    req2, timeout=HTTP_TIMEOUT_SECONDS, context=ctx_noverify
                ) as resp:
                    raw = resp.read()
                    encoding = resp.headers.get_content_charset() or "utf-8"
                    return raw.decode(encoding, errors="replace")

        raise URLError(f"Too many redirects ({_max_redirects}) for {url}")

    # ------------------------------------------------------------------
    # Timestamp normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_timestamp(ts: str) -> str:
        """Best-effort conversion of various timestamp formats to ISO 8601.

        Handles:
            - ISO 8601 (already valid, passthrough)
            - RFC 2822 (common in RSS: 'Mon, 03 Mar 2026 12:00:00 +0000')
            - Empty string (returns current UTC time)
        """
        if not ts:
            return datetime.now(timezone.utc).isoformat()

        # If already ISO-ish, return as-is
        if "T" in ts:
            return ts

        # Try RFC 2822 parsing (email.utils handles this well)
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(ts)
            return dt.isoformat()
        except Exception:
            pass

        # Try common date formats
        for fmt in [
            "%Y-%m-%d %H:%M:%S",
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S %Z",
            "%d %b %Y %H:%M:%S %z",
        ]:
            try:
                dt = datetime.strptime(ts, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat()
            except ValueError:
                continue

        # Last resort: return as-is
        return ts


# ---------------------------------------------------------------------------
# CLI test harness
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    print("=" * 70)
    print("Bull Machine NewsFetcher -- Standalone Test")
    print("=" * 70)

    fetcher = NewsFetcher()

    print("\n--- Latest Headlines (up to 10) ---\n")
    latest = fetcher.get_latest(count=10)

    if not latest:
        print("  (no items fetched -- check network or API key)")
    else:
        for i, item in enumerate(latest, 1):
            score_str = f"{item['sentiment_score']:+.2f}"
            print(
                f"  {i:2d}. [{item['sentiment']:>7s} {score_str}] "
                f"{item['headline'][:80]}"
            )
            print(f"      Source: {item['source']}  |  {item['published_at']}")
            if item["url"]:
                print(f"      URL: {item['url'][:90]}")
            print()

    print("--- Aggregate Sentiment ---\n")
    agg = fetcher.get_aggregate_sentiment()
    print(f"  Bullish:  {agg['bullish_count']}")
    print(f"  Bearish:  {agg['bearish_count']}")
    print(f"  Neutral:  {agg['neutral_count']}")
    print(f"  Avg Score: {agg['avg_score']:+.3f}")
    print(f"  Summary:  {agg['summary']}")

    print("\n" + "=" * 70)
    print("Test complete.")
