# tests/test_exchange_adapter.py
"""
Unit-Tests für exchange_adapter.py
Validates: Requirements 10.1, 10.2
"""

import inspect
import os
import sys
import unittest
from abc import ABC, abstractmethod
from unittest.mock import patch

# Ensure the project root is on sys.path so we can import exchange_adapter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exchange_adapter import (
    BitgetAdapter,
    ExchangeAdapter,
    get_adapter,
    _ADAPTERS,
)


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------

class TestGetAdapter(unittest.TestCase):
    """Test: get_adapter() returns correct BitgetAdapter instance."""

    def test_default_adapter_is_bitget(self):
        """get_adapter() returns a BitgetAdapter when EXCHANGE_ADAPTER='bitget'."""
        with patch("exchange_adapter.EXCHANGE_ADAPTER", "bitget"):
            adapter = get_adapter()
            self.assertIsInstance(adapter, BitgetAdapter)
            self.assertIsInstance(adapter, ExchangeAdapter)

    def test_unknown_adapter_raises_value_error(self):
        """get_adapter() raises ValueError with descriptive message for unknown adapter."""
        with patch("exchange_adapter.EXCHANGE_ADAPTER", "unknown_exchange"):
            with self.assertRaises(ValueError) as ctx:
                get_adapter()
            error_msg = str(ctx.exception)
            self.assertIn("unknown_exchange", error_msg)
            self.assertIn("bitget", error_msg)

    def test_adapter_registry_contains_bitget(self):
        """The adapter registry maps 'bitget' to BitgetAdapter."""
        self.assertIn("bitget", _ADAPTERS)
        self.assertIs(_ADAPTERS["bitget"], BitgetAdapter)


# ---------------------------------------------------------------------------
# Property 16: Exchange-Adapter Substitutionsprinzip
# Validates: Requirements 10.1, 10.2
#
# For any implementation of ExchangeAdapter, BitgetAdapter shall implement
# all abstract methods (get_candles, get_current_price, place_order,
# close_order, get_account_info, get_top_symbols).
# ---------------------------------------------------------------------------

class TestExchangeAdapterSubstitutionPrinzip(unittest.TestCase):
    """
    **Validates: Requirements 10.1, 10.2**

    Property 16: Exchange-Adapter Substitutionsprinzip —
    BitgetAdapter implements every abstract method declared in ExchangeAdapter.
    """

    # The full set of abstract methods that ExchangeAdapter declares.
    EXPECTED_ABSTRACT_METHODS = {
        "get_candles",
        "get_current_price",
        "place_order",
        "close_order",
        "get_account_info",
        "get_top_symbols",
    }

    def test_exchange_adapter_is_abstract(self):
        """ExchangeAdapter is an abstract base class and cannot be instantiated."""
        self.assertTrue(issubclass(ExchangeAdapter, ABC))
        with self.assertRaises(TypeError):
            ExchangeAdapter()

    def test_exchange_adapter_declares_expected_abstract_methods(self):
        """ExchangeAdapter declares exactly the expected set of abstract methods."""
        abstract_methods = set()
        for name, method in inspect.getmembers(ExchangeAdapter, predicate=inspect.isfunction):
            if getattr(method, "__isabstractmethod__", False):
                abstract_methods.add(name)
        self.assertEqual(abstract_methods, self.EXPECTED_ABSTRACT_METHODS)

    def test_bitget_adapter_implements_all_abstract_methods(self):
        """BitgetAdapter implements all abstract methods from ExchangeAdapter."""
        # If any abstract method is missing, BitgetAdapter itself would be
        # abstract and instantiation would raise TypeError.
        try:
            adapter = BitgetAdapter()
        except TypeError as exc:
            self.fail(
                f"BitgetAdapter konnte nicht instanziiert werden — "
                f"fehlende abstrakte Methoden: {exc}"
            )
        self.assertIsInstance(adapter, ExchangeAdapter)

    def test_bitget_adapter_methods_are_callable(self):
        """Every expected method on BitgetAdapter is callable."""
        adapter = BitgetAdapter()
        for method_name in self.EXPECTED_ABSTRACT_METHODS:
            self.assertTrue(
                callable(getattr(adapter, method_name, None)),
                f"BitgetAdapter.{method_name} is not callable",
            )

    def test_bitget_adapter_method_signatures_match(self):
        """BitgetAdapter method signatures match ExchangeAdapter declarations."""
        for method_name in self.EXPECTED_ABSTRACT_METHODS:
            base_sig = inspect.signature(getattr(ExchangeAdapter, method_name))
            impl_sig = inspect.signature(getattr(BitgetAdapter, method_name))
            self.assertEqual(
                list(base_sig.parameters.keys()),
                list(impl_sig.parameters.keys()),
                f"Signature mismatch for {method_name}: "
                f"base={list(base_sig.parameters.keys())} vs "
                f"impl={list(impl_sig.parameters.keys())}",
            )

    def test_bitget_adapter_has_no_remaining_abstract_methods(self):
        """BitgetAdapter has zero abstract methods remaining (fully concrete)."""
        remaining = getattr(BitgetAdapter, "__abstractmethods__", frozenset())
        self.assertEqual(
            remaining,
            frozenset(),
            f"BitgetAdapter still has abstract methods: {remaining}",
        )


if __name__ == "__main__":
    unittest.main()
