# Implementation Plan: ML-Meta-Strategie-Schicht für CryptoBot

## Overview

Schrittweiser Aufbau der ML-gestützten Meta-Strategie-Schicht als opt-in Gate zwischen der bestehenden trend-breakout-v3-Strategie und der Orderausführung. Die Implementierung beginnt mit Infrastruktur-Modulen (Adapter, Features, DB), gefolgt von der Kernlogik (Classifier, Gate, LLM-Advisor), dann Integration in die Trading-Loop und den Portfolio-Backtest, und endet mit Validierung und Fehlerresilienz-Tests. Programmiersprache: Python 3.12.

## Tasks

- [x] 1. Abhängigkeiten und Exchange-Adapter-Abstraktion
  - [x] 1.1 ML-Abhängigkeiten zu requirements.txt hinzufügen
    - `xgboost>=2.0.0`, `scikit-learn>=1.4.0`, `lightgbm>=4.0.0` ergänzen
    - `pip install -r requirements.txt` zur Verifikation ausführen
    - _Requirements: 3.1_

  - [x] 1.2 Exchange-Adapter-Abstraktion implementieren (`exchange_adapter.py`)
    - Abstrakte Basisklasse `ExchangeAdapter` mit Methoden: `get_candles`, `get_current_price`, `place_order`, `close_order`, `get_account_info`, `get_top_symbols`
    - Konkrete `BitgetAdapter`-Klasse, die an `bitget_client.py`-Funktionen delegiert
    - Factory-Funktion `get_adapter()` mit `EXCHANGE_ADAPTER`-Umgebungsvariable (Default: `"bitget"`)
    - Verifikation: `py_compile`, Import-Check, Prüfung dass `get_adapter()` eine `BitgetAdapter`-Instanz zurückgibt
    - _Requirements: 10.1, 10.2, 10.3, 10.4_

  - [x] 1.3 Unit-Tests für Exchange-Adapter schreiben
    - Test: `get_adapter()` gibt korrekte Instanz zurück
    - Test: Unbekannter Adapter-Name wirft `ValueError`
    - **Property 16: Exchange-Adapter Substitutionsprinzip** — alle abstrakten Methoden sind im `BitgetAdapter` implementiert
    - **Validates: Requirements 10.1, 10.2**

- [x] 2. Kausale Feature-Pipeline (`ml_features.py`)
  - [x] 2.1 Feature-Pipeline implementieren
    - `build_features(symbol, indicators)` für Live-Inferenz: lädt Candles via `fetch_ohlcv`, berechnet `prepare_indicators`, ergänzt abgeleitete Features (EMA-Spreads, ATR%, Donchian-Breite normalisiert, rollierende Volatilität, ADX-Slope, RSI-Momentum, DI-Spread, Volume-Ratio)
    - `build_features_from_frame(frame, index)` für Einzelzeilen-Extraktion aus vorbereiteten DataFrames (Training/Backtest)
    - `build_feature_matrix(df)` für Batch-Training: gibt DataFrame mit allen Feature-Spalten zurück
    - `FEATURE_COLUMNS`-Liste und `BASE_FEATURES`, `DERIVED_FEATURES` Konstanten
    - Alle Features kausal: nur Daten mit Zeitstempel ≤ t verwenden, `prepare_indicators()` und dessen `shift(1)`-Logik wiederverwenden
    - Verifikation: `py_compile`, Import-Check, Aufruf von `build_feature_matrix` mit einem synthetischen DataFrame prüft korrekte Spaltenzahl
    - _Requirements: 3.3, 6.1, 6.2_

  - [x] 2.2 Automatisierten Kausalitätstest implementieren (`causality_test`)
    - Methode: Werte in den letzten 5 Zeilen des DataFrames verdoppeln, prüfen dass Features bei Index -10 unverändert bleiben
    - Als Funktion `causality_test(df) -> bool` in `ml_features.py`
    - Verifikation: Aufruf mit einem synthetischen OHLCV-DataFrame, Ergebnis muss `True` sein
    - _Requirements: 6.4_

  - [x] 2.3 Property-Tests für Feature-Kausalität schreiben
    - **Property 6: Feature-Kausalität** — Manipulation von Daten an Zeitpunkten > t darf Features bei t nicht verändern
    - **Validates: Requirements 6.1, 6.2, 6.4**

- [x] 3. Checkpoint — Sicherstellen, dass Feature-Pipeline und Adapter korrekt kompilieren und importieren
  - Sicherstellen, dass alle Tests bestehen, den User fragen falls Unklarheiten auftreten.

- [x] 4. Datenbank-Erweiterung für ML-Vorhersagen (`ml_prediction_log.py`)
  - [x] 4.1 ML-Tabellen und Prediction-Logger implementieren
    - `init_ml_tables()`: Tabellen `ml_predictions` und `ml_model_metadata` anlegen via `CREATE TABLE IF NOT EXISTS`
    - Indizes für `(symbol, prediction_at)`, `(outcome_updated_at)`, `(symbol, is_active)`
    - `record(decision, symbol, top_features)` → persistiert eine ML-Vorhersage, gibt `prediction_id` zurück
    - `update_outcome(prediction_id, actual_regime, actual_performance)` → ergänzt tatsächliches Ergebnis
    - `query_accuracy(start_date, end_date)` → berechnet Vorhersagegenauigkeit, Block-Rate, Vergleich vorhergesagt/tatsächlich
    - Bestehende Tabellen (`trades`, `signals`, `funding_events`, `consumed_entry_signals`, `trade_quantity_epochs`) nicht modifizieren
    - Verifikation: `py_compile`, Import-Check, `init_ml_tables()` auf einer In-Memory-SQLite-DB aufrufen und Schema prüfen
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 14.1, 14.2, 14.3, 14.4_

  - [x] 4.2 `init_db()` in `database.py` erweitern
    - Am Ende von `init_db()`: `init_ml_tables()` aufrufen
    - `ALTER TABLE signals ADD COLUMN ml_prediction_id INTEGER` via bestehende `_cols`-Migration hinzufügen
    - Bestehende Tabellen-Definitionen unverändert lassen
    - Verifikation: `init_db()` auf frischer DB aufrufen, `ml_prediction_id`-Spalte in `signals` prüfen
    - _Requirements: 14.4, 14.5_

  - [x] 4.3 Unit-Tests für Prediction-Logger schreiben
    - Test: `record()` erzeugt korrekten Datensatz mit allen Pflichtfeldern
    - Test: `update_outcome()` ergänzt `actual_regime` und `actual_performance`
    - Test: `query_accuracy()` berechnet korrekte Genauigkeit
    - **Property 11: Vorhersage-Logging Vollständigkeit** — jede `GateDecision` erzeugt einen vollständigen Datensatz
    - **Property 14: Datenbank-Schema-Kompatibilität** — bestehende Tabellen bleiben unverändert, neue Tabellen idempotent
    - **Validates: Requirements 7.1, 7.4, 14.3, 14.4**

- [x] 5. Walk-Forward-Training und Regime-Klassifikator (`ml_trainer.py`)
  - [x] 5.1 Label-Generierung und Walk-Forward-Splits implementieren
    - `generate_labels(df, horizon, atr_threshold)`: Label = trending (1) wenn max. Preisbewegung in den folgenden N Candles > Schwellenwert * ATR, sonst choppy (0)
    - `walk_forward_splits(n_samples, train_ratio, n_splits)`: chronologisch geordnete Train/Val-Splits ohne Überlappung
    - `WalkForwardSplit`-Dataclass mit `train_start`, `train_end`, `val_start`, `val_end`
    - Konfiguration über `ML_LABEL_HORIZON` (Default 24) und `ML_LABEL_ATR_THRESHOLD` (Default 1.5)
    - Verifikation: `py_compile`, Test mit synthetischem DataFrame dass Labels und Splits korrekt generiert werden
    - _Requirements: 5.1, 5.7_

  - [x] 5.2 `RegimeClassifier`-Klasse implementieren
    - `predict(features)` → `{"regime": "trending"|"choppy", "probability": float}`
    - `load_latest(symbol, model_dir)` → lädt neuestes Modell für ein Symbol aus `model_dir`
    - `load_version(symbol, version, model_dir)` → lädt spezifische Modellversion für Rollback
    - `_load_from_dir(path)` → deserialisiert `model.pkl` und `metadata.json`
    - Verifikation: `py_compile`, Mock-Modell erstellen und `predict()`/`load_latest()` testen
    - _Requirements: 3.1, 3.4, 3.5, 12.4_

  - [x] 5.3 `WalkForwardTrainer`-Klasse implementieren
    - `train(symbol, df, feature_columns)`: XGBoost-Training auf Trainingsblock, Validierung auf Valblock, Metriken (Accuracy, Precision, Recall, F1, AUC-ROC) loggen
    - AUC-ROC-Regression verhindern: neues Modell ablehnen wenn schlechter als aktuelles (Anforderung 5.6)
    - `_save_model()`: Modell + Metadaten-JSON in `{ML_MODEL_DIR}/{symbol}_{timestamp}/` speichern (model.pkl, metadata.json mit Trainings-/Validierungszeitraum, Feature-Liste, Hyperparameter, Metriken, Daten-Hash)
    - `historical_predictions(symbol, df, feature_columns)`: gibt `[(timestamp, regime, confidence_score), ...]` für Backtest zurück
    - Historische Daten via `_load_ohlcv` aus `backtest.py` und `fetch_binance_funding_history` wiederverwenden
    - Metriken mit Präfix "historisch gemessen" oder "Backtest-Ergebnis" kennzeichnen
    - Verifikation: `py_compile`, Trainings-Durchlauf mit synthetischen Daten, prüfen dass Modelldateien und Metadaten korrekt gespeichert werden
    - _Requirements: 5.2, 5.3, 5.5, 5.6, 11.1, 12.1, 12.2, 13.1, 13.4_

  - [x] 5.4 Property-Tests für Walk-Forward-Training schreiben
    - **Property 7: Walk-Forward chronologische Trennung** — letzter Trainings-Zeitpunkt strikt vor erstem Validierungs-Zeitpunkt
    - **Property 12: Modell-AUC-Regression verhindert Deployment** — schlechteres Modell wird abgelehnt
    - **Property 13: Classifier-Inferenz-Wertebereich** — regime ∈ {"trending", "choppy"}, probability ∈ [0.0, 1.0]
    - **Property 15: Label-Generierung ausschließlich aus Zukunftsdaten** — Label bei Index i basiert nur auf Candles i+1..i+N
    - **Validates: Requirements 5.1, 5.6, 5.7, 6.3, 3.4**

- [x] 6. Checkpoint — Sicherstellen, dass Trainer, Classifier und Feature-Pipeline zusammenspielen
  - Sicherstellen, dass alle Tests bestehen, den User fragen falls Unklarheiten auftreten.

- [x] 7. ML-Gate Orchestrator (`ml_gate.py`)
  - [x] 7.1 Konfiguration und Validierung implementieren
    - Alle ML-Umgebungsvariablen lesen: `ML_GATE_ENABLED` (Default false), `ML_CONFIDENCE_THRESHOLD` (0.5), `ML_MIN_RISK_SCALE` (0.5), `ML_GB_WEIGHT` (0.7), `ML_LLM_WEIGHT` (0.3), `ML_LLM_ENABLED` (false), `ML_RETRAIN_INTERVAL_DAYS` (30), `ML_MODEL_DIR` (".models/")
    - `validate_config()`: ValueError bei ungültigen Werten (negative Schwellenwerte, Gewichte ≠ 1.0, Threshold außerhalb [0.0, 1.0])
    - `_log_config_banner()`: alle aktiven Konfigurationswerte beim Bot-Start loggen
    - Disclaimer-Hinweis loggen: "ML-Gate ist ein experimentelles Filter-System. Vergangene Backtest-Ergebnisse garantieren keine zukünftige Performance."
    - Verifikation: `py_compile`, `validate_config()` mit gültigen und ungültigen Werten testen
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 13.3_

  - [x] 7.2 Gate-Entscheidungslogik implementieren
    - `GateDecision`-Dataclass: `action`, `risk_scale`, `confidence`, `regime`, `filter_reason`, `gb_probability`, `llm_confidence`, `llm_regime`, `model_version`, `latency_ms`
    - `_BYPASS`-Konstante für deaktiviertes ML-Gate (action="allow", risk_scale=1.0)
    - `init(symbols)`: Modelle pro Symbol laden, bei Fehler Fallback-Modus
    - `evaluate(symbol, analysis)`: äußerer try/except gibt _BYPASS zurück bei Fehler; `_evaluate_inner()` baut Features, fragt Classifier und optional LLM-Advisor, berechnet gewichteten Confidence_Score
    - Gate-Logik: choppy + Confidence < Threshold → block; trending + Confidence ≥ Threshold → allow
    - `_compute_risk_scale(confidence)`: linearer Risk_Scale zwischen ML_MIN_RISK_SCALE und 1.0
    - `rollback_model(symbol, version)`: Rollback auf frühere Modellversion
    - Verifikation: `py_compile`, Import-Check, deterministischer Test mit Mock-Classifier
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.4, 3.5, 4.3, 8.4, 12.4, 15.1, 15.2, 15.3, 15.4_

  - [x] 7.3 Property-Tests für ML-Gate schreiben
    - **Property 1: ML-Gate-Bypass bei Deaktivierung** — bei `ML_GATE_ENABLED=false` immer action="allow", risk_scale=1.0
    - **Property 2: Choppy-Regime-Blockierung** — choppy + niedrige Confidence → block
    - **Property 3: Trending-Regime-Durchlass** — trending + hohe Confidence → allow
    - **Property 4: Risk-Scale-Bereichsinvariante** — risk_scale ∈ [ML_MIN_RISK_SCALE, 1.0], monoton steigend
    - **Property 8: Gewichtete Confidence-Score-Berechnung** — GB_WEIGHT * gb_prob + LLM_WEIGHT * llm_conf
    - **Property 9: Konfigurationsvalidierung** — ungültige Werte → ValueError
    - **Property 10: Fehlerresilienz — Exception-Isolation** — keine Exception erreicht den Aufrufer
    - **Validates: Requirements 1.2, 1.3, 1.4, 2.1, 2.4, 4.3, 4.4, 9.3, 15.1, 15.2, 15.4**

- [x] 8. Claude-LLM Regime-Advisor (`ml_gate_llm.py`)
  - [x] 8.1 LLM-Regime-Advisor implementieren
    - `LLMRegimeAdvisor`-Klasse mit `assess(symbol, indicators)` → `{"regime": str, "confidence": float}`
    - System-Prompt: Marktregime-Analyst, JSON-Antwort mit regime/confidence/reasoning
    - `_build_indicator_summary()`: nur aggregierte Indikatoren (ADX, RSI, EMAs, DI, ATR, Volume Ratio, MACD, Donchian, Price), keine Roh-OHLCV
    - `_call_claude()`: HTTP-POST an `api.anthropic.com/v1/messages` mit `CLAUDE_API_KEY` und konfiguriertem Timeout (`ML_LLM_TIMEOUT`, Default 10s)
    - `_parse_response()`: JSON parsen, Regime validieren (trending/choppy/neutral), Konfidenz auf [0.0, 1.0] clampen
    - Antwortzeit messen und loggen
    - Verifikation: `py_compile`, Import-Check, `_parse_response()` mit Mock-API-Antwort testen
    - _Requirements: 4.1, 4.2, 4.6_

  - [x] 8.2 Unit-Tests für LLM-Advisor schreiben
    - Test: `_build_indicator_summary()` enthält keine OHLCV-Rohdaten
    - Test: `_parse_response()` parst gültige und ungültige JSON-Antworten korrekt
    - Test: Timeout-Verhalten und Fallback
    - **Validates: Requirements 4.2, 4.4, 4.6**

- [x] 9. Integration in die Trading-Loop (`main.py`)
  - [x] 9.1 ML-Gate in `scan_new_entries()` integrieren
    - Nach `analyze_symbol(symbol)` und den bestehenden Signal-Prüfungen: `ml_gate.evaluate(symbol, analysis)` aufrufen
    - Bei `gate_decision.action == "block"`: Signal blockieren, loggen, `ml_prediction_log.record()` aufrufen, `continue`
    - Bei `gate_decision.action == "allow"`: `risk_scale = gate_decision.risk_scale` an `calculate_position_notional()` übergeben
    - `prediction_id = ml_prediction_log.record(gate_decision, symbol)` nach Order-Platzierung
    - Bei `ML_GATE_ENABLED=false`: `risk_scale=1.0` verwenden (Bypass)
    - Verifikation: `py_compile`, Import-Check, bestehende Kontrollfluss-Logik nicht verändern (MAX_OPEN_POSITIONS, daily_loss_limit_hit, exposure_headroom bleiben vorgelagert)
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.2, 2.3, 2.4, 8.1, 8.2, 8.3, 8.4, 8.5_

  - [x] 9.2 ML-Gate-Initialisierung in `main()` integrieren
    - Nach `init_db()`: `init_ml_tables()` aufrufen
    - Wenn `ML_GATE_ENABLED=true`: `ml_gate.init(symbols)` aufrufen
    - Verifikation: `py_compile`, Bot startet korrekt mit `ML_GATE_ENABLED=false`
    - _Requirements: 9.1, 12.3, 13.3_

  - [x] 9.3 Property-Test für bestehende Sicherheits-Limits schreiben
    - **Property 5: Bestehende Exposure-Limits als harte Obergrenze** — `calculate_position_notional()` überschreitet nie die Caps unabhängig vom `risk_scale`
    - **Validates: Requirements 2.3, 8.1**

- [x] 10. Checkpoint — Sicherstellen, dass die Trading-Loop mit ML-Gate kompiliert und die Bypass-Logik funktioniert
  - Sicherstellen, dass alle Tests bestehen, den User fragen falls Unklarheiten auftreten.

- [x] 11. Backtest-Integration
  - [x] 11.1 ML-Gate-Unterstützung in `portfolio_backtest.py` ergänzen
    - `SimParams` um `ml_gate_enabled: bool = False` und `ml_model_dir: str = ".models/"` erweitern
    - In `simulate()`: vor jeder Entry-Entscheidung historische Regime-Vorhersage via `WalkForwardTrainer.historical_predictions()` abfragen
    - Bei vorhergesagtem choppy-Regime und Confidence < Threshold: Entry blockieren, `skipped_filtered` zählen
    - `risk_scale` basierend auf historischem `Confidence_Score` an `_size_notional()` übergeben
    - Vergleichsausgabe: Performance-Metriken (Sharpe, Sortino, Max-Drawdown, Profit-Faktor, monatliche Renditen) mit und ohne ML-Gate nebeneinander
    - CLI-Argument `--ml-gate` zum Aktivieren der ML-Gate-Simulation
    - Historische Daten via bestehende `_load_ohlcv` und `fetch_binance_funding_history` aus `backtest.py` laden
    - Verifikation: `py_compile`, Backtest-Durchlauf mit und ohne `--ml-gate` Flag
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 13.1, 13.2_

  - [x] 11.2 Deterministische Tests für Backtest-Integration schreiben
    - Test: Backtest ohne ML-Gate produziert identische Ergebnisse wie bisher
    - Test: Backtest mit ML-Gate blockiert Entries bei choppy-Regime
    - Test: Vergleichsausgabe enthält beide Metrik-Sätze
    - **Validates: Requirements 11.2, 11.3, 11.4**

- [x] 12. Periodisches Retraining und Modell-Versionierung
  - [x] 12.1 Retraining-Logik in `ml_trainer.py` ergänzen
    - `should_retrain(symbol, model_dir)`: prüft ob seit letztem Training `ML_RETRAIN_INTERVAL_DAYS` vergangen sind (anhand `trained_at` in metadata.json)
    - In `ml_gate.init()`: nach dem Laden eines Modells prüfen ob Retraining fällig ist
    - Retraining-Ergebnis in `ml_model_metadata`-Tabelle persistieren
    - Verifikation: `py_compile`, Mock-basierter Test dass Retraining-Intervall korrekt geprüft wird
    - _Requirements: 5.4, 12.1, 12.2, 12.3_

  - [x] 12.2 Outcome-Tracking in der Trading-Loop implementieren
    - `ml_prediction_log.update_outcome()` nachträglich aufrufen wenn die Performance der folgenden N Candles verfügbar ist
    - Modellversion und Trainingszeitraum bei jeder Vorhersage annotieren
    - Verifikation: `py_compile`, Test dass `update_outcome()` den Datensatz korrekt ergänzt
    - _Requirements: 7.2, 7.4_

- [x] 13. Finale Validierung und Fehlerresilienz
  - [x] 13.1 Fehlerresilienz-Tests implementieren
    - Test: Classifier-Inferenz-Exception → Fallback (Confidence=0.5, Regime=trending, Risk_Scale=1.0)
    - Test: LLM-API-Timeout → nur GB-Wahrscheinlichkeit verwenden
    - Test: Modell nicht ladbar → Fallback-Modus (wie `ML_GATE_ENABLED=false`)
    - Test: Feature-Berechnung fehlerhaft → Bypass
    - Test: `ml_prediction_log.record()` Fehler → Trade fortsetzen
    - Test: Keine Exception aus `evaluate()` erreicht die Trading-Loop
    - Verifikation: Alle Tests bestehen
    - _Requirements: 15.1, 15.2, 15.3, 15.4_

  - [x] 13.2 End-to-End-Verifikation durchführen
    - `py_compile` für alle neuen Module: `ml_gate.py`, `ml_features.py`, `ml_trainer.py`, `ml_gate_llm.py`, `ml_prediction_log.py`, `exchange_adapter.py`
    - Import-Kette validieren: `main.py` importiert `ml_gate` und `ml_prediction_log` ohne Fehler
    - Bot-Start mit `ML_GATE_ENABLED=false`: Verhalten identisch zu ohne ML-Erweiterung
    - Bot-Start mit `ML_GATE_ENABLED=true` aber ohne trainiertes Modell: Fallback-Modus, Warnung im Log
    - Verifikation: Alle Compile-Checks bestehen
    - _Requirements: 1.4, 8.1, 8.2, 8.3, 8.4, 8.5, 9.1, 15.3_

- [x] 14. Finaler Checkpoint — Sicherstellen, dass alle Tests bestehen und Module korrekt zusammenspielen
  - Sicherstellen, dass alle Tests bestehen, den User fragen falls Unklarheiten auftreten.

## Notes

- Tasks mit `*` markiert sind optional und können für einen schnelleren MVP übersprungen werden
- Jeder Task referenziert spezifische Anforderungen für Nachvollziehbarkeit
- Checkpoints stellen inkrementelle Validierung sicher
- Property-Tests validieren universelle Korrektheitseigenschaften aus dem Design-Dokument
- Unit-Tests validieren spezifische Beispiele und Grenzfälle
- Die bestehende `risk_manager.py` hat bereits den `risk_scale`-Parameter in `calculate_position_notional()` — keine Änderung nötig
- Historische Daten für das Walk-Forward-Training werden über den Binance-Datenfetcher aus `backtest.py` (`_load_ohlcv`, `fetch_binance_funding_history`) geladen
- Der `portfolio_backtest.py` wird erweitert, nicht ersetzt — bestehende Validierungsergebnisse bleiben gültig
- Alle ML-Konfiguration lebt in `ml_gate.py`, nicht in `config.py` (Modul-Autonomie)

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["1.3", "2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "4.1"] },
    { "id": 3, "tasks": ["4.2", "4.3", "5.1"] },
    { "id": 4, "tasks": ["5.2", "5.3"] },
    { "id": 5, "tasks": ["5.4", "7.1"] },
    { "id": 6, "tasks": ["7.2", "8.1"] },
    { "id": 7, "tasks": ["7.3", "8.2"] },
    { "id": 8, "tasks": ["9.1", "9.2"] },
    { "id": 9, "tasks": ["9.3", "11.1"] },
    { "id": 10, "tasks": ["11.2", "12.1"] },
    { "id": 11, "tasks": ["12.2", "13.1"] },
    { "id": 12, "tasks": ["13.2"] }
  ]
}
```
