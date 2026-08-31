# ml_gate_llm.py

import json
import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
ML_LLM_TIMEOUT = int(os.getenv("ML_LLM_TIMEOUT", "10"))


class LLMRegimeAdvisor:
    """
    Anforderung 4: Qualitative Markteinschätzung über Claude-API.
    Übergibt ausschließlich aggregierte Indikator-Zusammenfassungen.
    """

    SYSTEM_PROMPT = (
        "Du bist ein Marktregime-Analyst. Analysiere die folgenden technischen "
        "Indikatoren und klassifiziere das Marktregime als trending, choppy "
        "oder neutral. Gib eine Konfidenz zwischen 0.0 und 1.0 an. "
        "Antworte ausschließlich im JSON-Format: "
        '{"regime": "trending|choppy|neutral", "confidence": 0.0-1.0, '
        '"reasoning": "kurze Begründung"}'
    )

    def assess(self, symbol: str, indicators: dict) -> dict:
        """
        Anforderung 4.1, 4.2: Nur aggregierte Indikatoren, keine Roh-OHLCV.
        Anforderung 4.6: Antwortzeit messen und protokollieren.
        """
        summary = self._build_indicator_summary(symbol, indicators)

        start = time.monotonic()
        try:
            response = self._call_claude(summary)
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.info(f"LLM-Advisor: {symbol} Antwortzeit={elapsed_ms:.0f}ms")

        return self._parse_response(response)

    def _build_indicator_summary(self, symbol: str, indicators: dict) -> str:
        """Anforderung 4.2: Nur aggregierte Zusammenfassungen."""
        return (
            f"Symbol: {symbol}\n"
            f"ADX: {indicators.get('adx', 'N/A')}\n"
            f"RSI: {indicators.get('rsi', 'N/A')}\n"
            f"EMA20: {indicators.get('ema20', 'N/A')}\n"
            f"EMA50: {indicators.get('ema50', 'N/A')}\n"
            f"EMA200: {indicators.get('ema200', 'N/A')}\n"
            f"DI+: {indicators.get('di_pos', 'N/A')}\n"
            f"DI-: {indicators.get('di_neg', 'N/A')}\n"
            f"ATR: {indicators.get('atr', 'N/A')}\n"
            f"Volume Ratio: {indicators.get('volume_ratio', 'N/A')}\n"
            f"MACD Histogram: {indicators.get('macd_hist', 'N/A')}\n"
            f"Donchian High: {indicators.get('donchian_high', 'N/A')}\n"
            f"Donchian Low: {indicators.get('donchian_low', 'N/A')}\n"
            f"Current Price: {indicators.get('current_price', 'N/A')}"
        )

    def _call_claude(self, summary: str) -> dict:
        """HTTP-Aufruf an die Claude-API."""
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": CLAUDE_API_KEY,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 256,
                "system": self.SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": summary}],
            },
            timeout=ML_LLM_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()

    def _parse_response(self, api_response: dict) -> dict:
        """Parst die Claude-Antwort und extrahiert Regime + Konfidenz."""
        content = api_response.get("content", [{}])
        text = content[0].get("text", "{}") if content else "{}"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            raise ValueError(f"Ungültige LLM-Antwort: {text[:200]}")

        regime = parsed.get("regime", "neutral")
        if regime not in ("trending", "choppy", "neutral"):
            regime = "neutral"
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
        return {"regime": regime, "confidence": confidence}
