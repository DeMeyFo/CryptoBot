# exchange_adapter.py

import logging
import os
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ExchangeAdapter(ABC):
    """
    Anforderung 10.1: Abstrakte Schnittstelle für Marktdaten und Orderausführung.
    Ermöglicht spätere Erweiterung auf Forex/MT5.
    """

    @abstractmethod
    def get_candles(self, symbol: str, granularity: str, limit: int) -> list:
        ...

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        ...

    @abstractmethod
    def place_order(self, symbol: str, side: str, size_usdt: float,
                    leverage: int, current_price: float) -> dict:
        ...

    @abstractmethod
    def close_order(self, symbol: str, side: str, quantity: float,
                    observed_price: float) -> dict:
        ...

    @abstractmethod
    def get_account_info(self) -> dict:
        ...

    @abstractmethod
    def get_top_symbols(self, count: int) -> list[str]:
        ...


class BitgetAdapter(ExchangeAdapter):
    """
    Anforderung 10.2: Konkrete Implementierung für Bitget.
    Kapselt die bestehenden Funktionen aus bitget_client.py.
    """

    def get_candles(self, symbol: str, granularity: str, limit: int) -> list:
        from bitget_client import get_candles
        return get_candles(symbol, granularity, limit)

    def get_current_price(self, symbol: str) -> float:
        from bitget_client import get_current_price
        return get_current_price(symbol)

    def place_order(self, symbol: str, side: str, size_usdt: float,
                    leverage: int, current_price: float) -> dict:
        from bitget_client import place_order
        return place_order(symbol, side, size_usdt, leverage, current_price)

    def close_order(self, symbol: str, side: str, quantity: float,
                    observed_price: float) -> dict:
        from bitget_client import close_order
        return close_order(symbol, side, quantity, observed_price)

    def get_account_info(self) -> dict:
        from bitget_client import get_account_info
        return get_account_info()

    def get_top_symbols(self, count: int) -> list[str]:
        from bitget_client import get_top_symbols
        return get_top_symbols(count)


# Anforderung 10.4: Adapter-Auswahl via Umgebungsvariable
EXCHANGE_ADAPTER = os.getenv("EXCHANGE_ADAPTER", "bitget")

_ADAPTERS = {
    "bitget": BitgetAdapter,
}


def get_adapter() -> ExchangeAdapter:
    """Factory: Erzeugt den konfigurierten Exchange-Adapter."""
    adapter_class = _ADAPTERS.get(EXCHANGE_ADAPTER)
    if adapter_class is None:
        raise ValueError(
            f"Unbekannter EXCHANGE_ADAPTER: {EXCHANGE_ADAPTER}. "
            f"Verfügbar: {sorted(_ADAPTERS.keys())}"
        )
    return adapter_class()
