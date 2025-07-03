# defillama_utils.py

import requests
import datetime
from typing import List, Dict, Tuple, Optional

"""defillama_utils.py — lightweight helpers that pull free on‑chain fundamentals
from DeFi Llama’s open API so your agent can issue Buy / Sell / Hold
signals without paid data feeds.

Public API docs: https://defillama.com/docs/api
All endpoints used here require **no API key**.
"""

BASE_URL = "https://api.llama.fi"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_json(endpoint: str) -> Dict:
    """GET a DeFi Llama endpoint and return its JSON body.
    Raises requests.HTTPError on 4xx / 5xx.
    """
    resp = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
    resp.raise_for_status()
    return resp.json()


def _get_protocols() -> List[Dict]:
    """Return the full list of tracked protocols (cached by caller)."""
    return _fetch_json("/protocols")


def _find_slug(symbol: str) -> Tuple[Optional[str], Optional[str]]:
    """Map a **token symbol** to DeFi Llama *slug* and *name*.

    If the symbol represents a *base chain* (e.g. SOL, ETH) there may be no
    slug; in that case the caller should treat the symbol as a chain name and
    use chain‑level endpoints.
    """
    symbol = symbol.upper()
    for proto in _get_protocols():
        proto_symbol = proto.get("symbol")
        if not proto_symbol:
            continue
        # symbol can be string or list in API response
        symbols = (
            [s.upper() for s in proto_symbol]
            if isinstance(proto_symbol, list)
            else [proto_symbol.upper()]
        )
        if symbol in symbols:
            return proto.get("slug"), proto.get("name")
    return None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_fundamentals(symbol: str, lookback_days: int = 30) -> str:
    """Return a markdown summary of free fundamentals for *symbol*.

    The summary includes:
      • Latest TVL (total value locked)
      • % change in TVL over *lookback_days*
      • Sum of protocol FEES and REVENUE over *lookback_days*

    Args:
        symbol: Token ticker such as 'UNI', 'SOL', 'GMX'.
        lookback_days: Window size for change / sum calculations.

    Returns:
        Markdown‑formatted string suitable for LLM prompts or dashboards.
    """

    slug, nice_name = _find_slug(symbol)
    if not slug:
        return f"Symbol '{symbol}' not found on DeFi Llama."

    # ---------------- TVL ----------------
    try:
        proto_json = _fetch_json(f"/protocol/{slug}")
        tvl_series = sorted(proto_json.get("tvl", []), key=lambda d: d["date"])
    except Exception as exc:
        return f"Error fetching TVL data: {exc}"

    if not tvl_series:
        return f"No TVL data available for '{symbol}'."

    latest_entry = tvl_series[-1]
    latest_ts: int = latest_entry["date"]
    latest_tvl: float = latest_entry.get("totalLiquidityUSD", 0.0)

    cutoff_ts = latest_ts - lookback_days * 86_400  # seconds in a day
    past_points = [p for p in tvl_series if p["date"] <= cutoff_ts]
    past_tvl = past_points[-1]["totalLiquidityUSD"] if past_points else None
    tvl_pct = ((latest_tvl - past_tvl) / past_tvl * 100) if past_tvl else None

    # ---------------- Fees / Revenue ----------------
    fees_sum = rev_sum = None
    try:
        fees_json = _fetch_json(f"/summary/fees/{slug}")
        fees_chart = fees_json.get("totalDataChart", [])
        rev_chart = fees_json.get("revenueDataChart", [])

        def _sum_last(chart):
            if not chart:
                return None
            cutoff = chart[-1][0] - lookback_days * 86_400
            return sum(point[1] for point in chart if point[0] >= cutoff)

        fees_sum = _sum_last(fees_chart)
        rev_sum = _sum_last(rev_chart)
    except Exception:
        # Some protocols or chains won’t have fee data; ignore.
        pass

    latest_date = datetime.datetime.utcfromtimestamp(latest_ts).strftime("%Y-%m-%d")

    # ---------------- Assemble markdown ----------------
    lines = [
        f"### {nice_name or slug.upper()} Fundamentals (as of {latest_date})\n",
        f"- **Latest TVL:** ${latest_tvl:,.0f}",
    ]
    if tvl_pct is not None:
        lines.append(f"- **TVL Δ {lookback_days}d:** {tvl_pct:+.2f}%")
    if fees_sum is not None:
        lines.append(f"- **Fees collected ({lookback_days}d):** ${fees_sum:,.0f}")
    if rev_sum is not None:
        lines.append(f"- **Revenue ({lookback_days}d):** ${rev_sum:,.0f}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick‑check (only runs when executed directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(get_fundamentals("UNI"))






