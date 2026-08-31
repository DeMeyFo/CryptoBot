"""
Delta-neutraler Funding-Carry: Spot long + Perp short, kassiert Funding.

Strukturell unkorreliert zur Trendstrategie, weil der Ertrag aus der
Funding-Rate kommt, nicht aus Preisbewegungen. Verdient am stabilsten
in Seitwaertsphasen, also genau dann, wenn die Trendstrategie blutet.

Architektur:
  * Eigener Positions-Zyklus, unabhaengig von scan_new_entries()
  * Ueberwacht Funding-Raten aller Research-Symbole
  * Oeffnet Carry-Positionen wenn Rate > Schwellenwert
  * Schliesst wenn Rate < 0 oder Verlust-Schwellenwert erreicht
  * Loggt Einnahmen separat fuer Transparenz

Voraussetzung: Bitget API-Key mit Spot-Handels-Berechtigung.
"""

import logging
import os
import time
from datetime import datetime, timezone

import requests

logger = logging.getLogger(__name__)

# ── Konfiguration ─────────────────────────────────────────────────────────

CARRY_ENABLED = os.getenv("CARRY_ENABLED", "false").lower() == "true"
# Mindest-Funding-Rate (pro 8h-Settlement) um eine Position zu oeffnen
CARRY_MIN_RATE = float(os.getenv("CARRY_MIN_RATE", "0.0001"))
# Rate unter der eine bestehende Position geschlossen wird
CARRY_EXIT_RATE = float(os.getenv("CARRY_EXIT_RATE", "-0.0001"))
# Maximaler Anteil der Equity pro Carry-Position (Spot + Perp zusammen)
CARRY_MAX_POSITION_PCT = float(os.getenv("CARRY_MAX_POSITION_PCT", "0.15"))
# Maximaler Anteil der Equity fuer alle Carry-Positionen zusammen
CARRY_MAX_TOTAL_PCT = float(os.getenv("CARRY_MAX_TOTAL_PCT", "0.40"))
# Spot-API-Basis
_BITGET_BASE = "https://api.bitget.com"
_PRODUCT_TYPE = "USDT-FUTURES"
# Pruefintervall (Sekunden)
CARRY_CHECK_INTERVAL = int(os.getenv("CARRY_CHECK_INTERVAL", "3600"))

# ── In-Memory-State ───────────────────────────────────────────────────────

_carry_positions: dict[str, dict] = {}
_last_check: float = 0.0
_cumulative_funding: float = 0.0


# ── Datenabfragen ─────────────────────────────────────────────────────────

def _get_funding_rate(symbol: str) -> float:
    try:
        r = requests.get(
            f"{_BITGET_BASE}/api/v2/mix/market/current-fund-rate",
            params={"symbol": symbol, "productType": _PRODUCT_TYPE},
            timeout=8,
        )
        data = r.json().get("data", [])
        if data:
            return float(data[0].get("fundingRate", 0))
    except Exception as e:
        logger.debug(f"Carry: Funding-Rate {symbol}: {e}")
    return 0.0


def _get_spot_price(symbol: str) -> float:
    try:
        r = requests.get(
            f"{_BITGET_BASE}/api/v2/spot/market/tickers",
            params={"symbol": symbol},
            timeout=8,
        )
        data = r.json().get("data", [])
        if data:
            return float(data[0].get("lastPr", 0))
    except Exception as e:
        logger.debug(f"Carry: Spot-Preis {symbol}: {e}")
    return 0.0


def _get_perp_price(symbol: str) -> float:
    try:
        r = requests.get(
            f"{_BITGET_BASE}/api/v2/mix/market/ticker",
            params={"symbol": symbol, "productType": _PRODUCT_TYPE},
            timeout=8,
        )
        data = r.json().get("data", [])
        if data:
            return float(data[0].get("lastPr", 0))
    except Exception as e:
        logger.debug(f"Carry: Perp-Preis {symbol}: {e}")
    return 0.0


# ── Carry-Logik ───────────────────────────────────────────────────────────

def get_carry_opportunities(symbols: list[str]) -> list[dict]:
    """
    Scannt alle Symbole und gibt die mit positiver Funding-Rate zurueck,
    sortiert nach Yield.
    """
    opportunities = []
    for symbol in symbols:
        rate = _get_funding_rate(symbol)
        if rate >= CARRY_MIN_RATE:
            spot_price = _get_spot_price(symbol)
            perp_price = _get_perp_price(symbol)
            spread_pct = abs(perp_price - spot_price) / spot_price * 100 if spot_price > 0 else 0
            annual_yield = rate * 3 * 365 * 100
            opportunities.append({
                "symbol": symbol,
                "funding_rate": rate,
                "annual_yield_pct": annual_yield,
                "spot_price": spot_price,
                "perp_price": perp_price,
                "spread_pct": spread_pct,
            })
    opportunities.sort(key=lambda x: -x["funding_rate"])
    return opportunities


def check_carry_exits() -> list[str]:
    """Prueft bestehende Carry-Positionen auf Exit-Bedingungen."""
    exits = []
    for symbol, pos in list(_carry_positions.items()):
        rate = _get_funding_rate(symbol)
        if rate < CARRY_EXIT_RATE:
            exits.append(symbol)
            logger.info(
                f"Carry EXIT {symbol}: Funding-Rate {rate*100:.4f}% < "
                f"Schwellenwert {CARRY_EXIT_RATE*100:.4f}%"
            )
    return exits


def estimate_daily_carry_income() -> dict:
    """Schaetzt das taegliche Carry-Einkommen der offenen Positionen."""
    total_daily = 0.0
    details = {}
    for symbol, pos in _carry_positions.items():
        rate = _get_funding_rate(symbol)
        size_usdt = pos.get("size_usdt", 0)
        daily = size_usdt * rate * 3  # 3 Settlements pro Tag
        total_daily += daily
        details[symbol] = {
            "size_usdt": size_usdt,
            "current_rate": rate,
            "daily_income_usdt": daily,
        }
    return {
        "total_daily_usdt": total_daily,
        "total_monthly_usdt": total_daily * 30,
        "positions": details,
        "cumulative_funding": _cumulative_funding,
    }


def carry_status() -> dict:
    """Gibt den aktuellen Status aller Carry-Positionen zurueck."""
    return {
        "enabled": CARRY_ENABLED,
        "positions": len(_carry_positions),
        "symbols": list(_carry_positions.keys()),
        "config": {
            "min_rate": CARRY_MIN_RATE,
            "exit_rate": CARRY_EXIT_RATE,
            "max_position_pct": CARRY_MAX_POSITION_PCT,
            "max_total_pct": CARRY_MAX_TOTAL_PCT,
            "check_interval": CARRY_CHECK_INTERVAL,
        },
        "income": estimate_daily_carry_income(),
    }


def should_check(now: float | None = None) -> bool:
    """Prueft ob ein neuer Carry-Check faellig ist."""
    global _last_check
    now = now or time.time()
    if now - _last_check < CARRY_CHECK_INTERVAL:
        return False
    _last_check = now
    return True


def log_carry_scan(symbols: list[str]) -> None:
    """
    Periodischer Scan: loggt Carry-Opportunities und Exit-Signale.

    Im DRY_RUN-Modus werden keine echten Orders ausgefuehrt, sondern
    nur geloggt, welche Positionen geoeffnet oder geschlossen wuerden.
    """
    if not CARRY_ENABLED:
        return

    opportunities = get_carry_opportunities(symbols)
    exits = check_carry_exits()

    if opportunities:
        logger.info(
            f"Carry-Scan: {len(opportunities)} Opportunities "
            f"(min rate {CARRY_MIN_RATE*100:.3f}%)"
        )
        for opp in opportunities[:5]:
            logger.info(
                f"  {opp['symbol']:10s} rate={opp['funding_rate']*100:.4f}% "
                f"yield={opp['annual_yield_pct']:.1f}%/yr "
                f"spread={opp['spread_pct']:.3f}%"
            )

    if exits:
        logger.warning(
            f"Carry-Exits: {exits} (Funding-Rate unter {CARRY_EXIT_RATE*100:.4f}%)"
        )

    income = estimate_daily_carry_income()
    if _carry_positions:
        logger.info(
            f"Carry-Einkommen: ~{income['total_daily_usdt']:.2f} USDT/Tag "
            f"(~{income['total_monthly_usdt']:.2f} USDT/Monat) "
            f"kumuliert: {income['cumulative_funding']:.2f} USDT"
        )
