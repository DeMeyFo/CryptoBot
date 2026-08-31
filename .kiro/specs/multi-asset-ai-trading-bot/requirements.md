# Requirements Document

## Introduction

Dieses Dokument beschreibt die Anforderungen für eine ML-gestützte Meta-Strategie-Schicht, die den bestehenden CryptoBot um maschinelles Lernen erweitert. Das System fungiert als intelligenter Gatekeeper: Es entscheidet, WANN die validierte trend-breakout-v3-Strategie handeln darf, und skaliert Positionsgrößen basierend auf ML-Konfidenz. Die Kernstrategie-Logik (Donchian-55 Breakout + ADX≥40 + EMA-Struktur + RSI + DI) bleibt unverändert. Der ML-Layer kombiniert Gradient Boosting (XGBoost/LightGBM) für taktische Regime-Erkennung mit Claude LLM für qualitative Markteinschätzung. Die Architektur erlaubt spätere Erweiterung auf Forex/MT5.

## Glossary

- **ML_Gate**: Das maschinelle Lernmodul, das Regime-Vorhersagen und Konfidenzwerte liefert, um Entry-Entscheidungen der Kernstrategie zu filtern und Positionsgrößen zu skalieren.
- **Regime_Classifier**: Die Gradient-Boosting-Komponente (XGBoost oder LightGBM), die historische Indikator- und Marktdaten nutzt, um das aktuelle Marktregime als trending oder choppy zu klassifizieren.
- **LLM_Regime_Advisor**: Die Claude-LLM-Komponente, die qualitative Marktbedingungen analysiert und eine Regime-Einschätzung mit Konfidenz liefert.
- **Confidence_Score**: Ein numerischer Wert zwischen 0.0 und 1.0, der die gewichtete Kombination aus Regime_Classifier und LLM_Regime_Advisor darstellt.
- **Feature_Pipeline**: Das Modul, das aus OHLCV-Daten, technischen Indikatoren und Marktstruktur-Metriken kausal korrekte Features für das Training und die Inferenz des Regime_Classifier berechnet.
- **Walk_Forward_Trainer**: Das Modul, das den Regime_Classifier nach einem Walk-Forward-Protokoll trainiert: Suche auf einem Block, Validierung auf einem getrennten Folge-Block.
- **Prediction_Logger**: Das Modul, das alle ML-Vorhersagen mit Zeitstempel, Features, Konfidenz und tatsächlichem Ergebnis persistiert.
- **Risk_Manager**: Das bestehende Risikomanagement-Modul (risk_manager.py), das Stop-Loss, Trailing-Stop, Exposure-Limits und Positionsgrößenberechnung steuert.
- **Core_Strategy**: Die bestehende trend-breakout-v3-Strategie in strategy.py und technical_analysis.py.
- **Trading_Loop**: Die Hauptschleife in main.py, die Entry-Scans, Trailing-Stops und Position-Management orchestriert.
- **Causal_Feature**: Ein Feature, dessen Berechnung zum Zeitpunkt t ausschließlich Daten aus Zeitpunkten ≤ t verwendet (kein Look-Ahead-Bias).
- **Risk_Scale**: Der Skalierungsfaktor (0.0–1.0), mit dem die ML-Konfidenz die Positionsgröße innerhalb der bestehenden Risk_Manager-Limits moduliert.
- **Exchange_Adapter**: Eine abstrakte Schnittstellendefinition, die den Zugriff auf Marktdaten und Orderausführung kapselt, um spätere Erweiterung auf andere Börsen (z.B. MT5 für Forex) zu ermöglichen.

## Requirements

### Anforderung 1: ML-Gate Entry-Filterung

**User Story:** Als Trader möchte ich, dass ein ML-Modell die bestehenden Breakout-Signale vor der Orderausführung filtert, damit Entries in choppy Regimen unterdrückt werden und nur in vorhergesagten Trend-Regimen gehandelt wird.

#### Akzeptanzkriterien

1. WHEN die Core_Strategy ein Entry-Signal (long oder short) erzeugt, THE ML_Gate SHALL eine Regime-Vorhersage und einen Confidence_Score für das betreffende Symbol berechnen, bevor die Entry-Entscheidung an den Risk_Manager weitergegeben wird.
2. WHEN der Regime_Classifier das aktuelle Regime als choppy klassifiziert UND der Confidence_Score unter einem konfigurierbaren Schwellenwert liegt, THE ML_Gate SHALL das Entry-Signal blockieren und die Aktion als "hold" mit dem Filtergrund "ml_regime_blocked" protokollieren.
3. WHEN der Regime_Classifier das aktuelle Regime als trending klassifiziert UND der Confidence_Score über dem konfigurierbaren Schwellenwert liegt, THE ML_Gate SHALL das Entry-Signal unverändert an den Risk_Manager weiterleiten.
4. WHILE die Umgebungsvariable `ML_GATE_ENABLED` auf "false" gesetzt ist, THE ML_Gate SHALL alle Entry-Signale der Core_Strategy ohne Filterung durchlassen, sodass die trend-breakout-v3-Strategie identisch wie ohne ML-Erweiterung funktioniert.
5. THE ML_Gate SHALL den konfigurierbaren Regime-Schwellenwert über die Umgebungsvariable `ML_CONFIDENCE_THRESHOLD` mit einem Standardwert von 0.5 lesen.

### Anforderung 2: ML-basierte Positionsgrößenskalierung

**User Story:** Als Trader möchte ich, dass die Positionsgröße basierend auf der ML-Konfidenz skaliert wird, damit bei niedriger Konfidenz kleinere und bei hoher Konfidenz größere Positionen eröffnet werden.

#### Akzeptanzkriterien

1. WHEN der ML_Gate ein Entry-Signal durchlässt, THE ML_Gate SHALL einen Risk_Scale-Faktor zwischen einem konfigurierbaren Minimum (Standardwert 0.5) und 1.0 berechnen, proportional zum Confidence_Score.
2. THE Trading_Loop SHALL den Risk_Scale-Faktor an die bestehende Funktion `calculate_position_notional` im Parameter `risk_scale` übergeben.
3. THE Risk_Manager SHALL die bestehenden Exposure-Limits (MAX_SYMBOL_EXPOSURE_PCT, MAX_GROSS_EXPOSURE_PCT), die Positionsgrößen-Caps (MAX_POSITION_PCT, MAX_POSITION_USDT) und die Floor-Werte (MIN_POSITION_PCT, MIN_POSITION_USDT) unverändert als harte Obergrenzen beibehalten, unabhängig vom Risk_Scale-Wert.
4. WHILE die Umgebungsvariable `ML_GATE_ENABLED` auf "false" gesetzt ist, THE Trading_Loop SHALL den Risk_Scale-Wert 1.0 verwenden.

### Anforderung 3: Gradient-Boosting Regime-Klassifikator

**User Story:** Als Trader möchte ich, dass ein Gradient-Boosting-Modell anhand historischer Indikator- und Marktdaten das Marktregime erkennt, damit die Entry-Filterung auf quantitativen Signalen basiert.

#### Akzeptanzkriterien

1. THE Regime_Classifier SHALL als XGBoost- oder LightGBM-Binärklassifikator implementiert werden, der das Marktregime pro Symbol und Zeitfenster als trending (1) oder choppy (0) klassifiziert.
2. THE Regime_Classifier SHALL ausschließlich Causal_Features verwenden, die zum Inferenzzeitpunkt aus bereits abgeschlossenen Candles berechnet wurden.
3. THE Feature_Pipeline SHALL die folgenden Feature-Kategorien bereitstellen: technische Indikatoren aus prepare_indicators (ADX, RSI, EMA-Spreads, ATR, Donchian-Breiten, Volumen-Ratio, MACD-Histogramm, DI-Spread), abgeleitete Features (rollierende Volatilität, ADX-Trend, RSI-Momentum) und Marktstruktur-Metriken (ATR als Prozent des Preises, Donchian-Kanalbreite normalisiert).
4. THE Regime_Classifier SHALL bei der Inferenz eine Klassifikation (trending oder choppy) und einen Wahrscheinlichkeitswert zwischen 0.0 und 1.0 zurückgeben.
5. IF der Regime_Classifier kein trainiertes Modell laden kann, THEN THE ML_Gate SHALL auf den Standardwert Confidence_Score=0.5 und Regime=trending zurückfallen und eine Warnung protokollieren.

### Anforderung 4: Claude-LLM Regime-Advisor

**User Story:** Als Trader möchte ich, dass ein LLM qualitative Markteinschätzungen liefert, damit die Regime-Erkennung um Kontext ergänzt wird, den ein rein quantitatives Modell nicht erfassen kann.

#### Akzeptanzkriterien

1. WHEN ein Entry-Signal durch die Core_Strategy erzeugt wird UND die Umgebungsvariable `ML_LLM_ENABLED` auf "true" gesetzt ist, THE LLM_Regime_Advisor SHALL eine Regime-Einschätzung (trending, choppy oder neutral) und einen Konfidenzwert zwischen 0.0 und 1.0 von der Claude-API abrufen.
2. THE LLM_Regime_Advisor SHALL dem LLM ausschließlich aggregierte Indikator-Zusammenfassungen und keine Roh-OHLCV-Daten übergeben.
3. THE ML_Gate SHALL den finalen Confidence_Score als gewichtete Kombination aus dem Regime_Classifier-Wahrscheinlichkeitswert und dem LLM_Regime_Advisor-Konfidenzwert berechnen, wobei die Gewichte über die Umgebungsvariablen `ML_GB_WEIGHT` (Standardwert 0.7) und `ML_LLM_WEIGHT` (Standardwert 0.3) konfigurierbar sind.
4. IF die Claude-API nicht erreichbar ist oder einen Fehler zurückgibt, THEN THE ML_Gate SHALL ausschließlich den Regime_Classifier-Wahrscheinlichkeitswert als Confidence_Score verwenden und eine Warnung protokollieren.
5. WHILE die Umgebungsvariable `ML_LLM_ENABLED` auf "false" gesetzt ist (Standardwert), THE ML_Gate SHALL ausschließlich den Regime_Classifier-Wahrscheinlichkeitswert als Confidence_Score verwenden.
6. THE LLM_Regime_Advisor SHALL die Antwortzeit jedes API-Aufrufs messen und protokollieren.

### Anforderung 5: Walk-Forward-Training

**User Story:** Als Trader möchte ich, dass das ML-Modell nach einem Walk-Forward-Protokoll trainiert wird, damit Overfitting vermieden und die Generalisierbarkeit des Modells sichergestellt wird.

#### Akzeptanzkriterien

1. THE Walk_Forward_Trainer SHALL die verfügbaren historischen Daten in chronologische Blöcke aufteilen: einen Trainingsblock für die Modellsuche und einen zeitlich nachfolgenden, nicht überlappenden Validierungsblock für die Leistungsbewertung.
2. THE Walk_Forward_Trainer SHALL die Hyperparameter-Suche ausschließlich auf dem Trainingsblock durchführen und die gefundenen Parameter auf dem Validierungsblock evaluieren.
3. THE Walk_Forward_Trainer SHALL das Modell als serialisierte Datei im Verzeichnis `.models/` persistieren, wobei der Dateiname den Trainingszeitraum und einen Versionsbezeichner enthält.
4. THE Walk_Forward_Trainer SHALL periodisch nach einem konfigurierbaren Zeitplan (Umgebungsvariable `ML_RETRAIN_INTERVAL_DAYS`, Standardwert 30) einen neuen Walk-Forward-Zyklus ausführen.
5. THE Walk_Forward_Trainer SHALL nach jedem Trainingszyklus die Validierungsmetriken (Accuracy, Precision, Recall, F1-Score, AUC-ROC) protokollieren.
6. IF ein neu trainiertes Modell auf dem Validierungsblock eine schlechtere AUC-ROC erzielt als das aktuell eingesetzte Modell, THEN THE Walk_Forward_Trainer SHALL das bestehende Modell beibehalten und eine Warnung protokollieren.
7. THE Walk_Forward_Trainer SHALL bei der Label-Generierung für das Regime die realisierte Performance der folgenden N Candles (konfigurierbar, Standardwert 24) auswerten: trending, wenn die Preisbewegung einen konfigurierbaren ATR-Schwellenwert überschreitet, choppy andernfalls.

### Anforderung 6: Kausalitäts-Garantie der Features

**User Story:** Als Trader möchte ich sicherstellen, dass keine Features zukunftsbezogene Informationen enthalten, damit die Backtest-Ergebnisse und Live-Performance vergleichbar sind.

#### Akzeptanzkriterien

1. THE Feature_Pipeline SHALL jedes Feature ausschließlich aus Daten berechnen, deren Zeitstempel kleiner oder gleich dem Zeitstempel der aktuell verarbeiteten Candle ist.
2. THE Feature_Pipeline SHALL die bestehende `prepare_indicators`-Funktion aus technical_analysis.py wiederverwenden und die gleiche `shift(1)`-Logik für Donchian-Levels und die gleiche `_finite_ema`-Funktion für EMA-Berechnung anwenden.
3. THE Walk_Forward_Trainer SHALL die Trainings-/Validierungsblöcke strikt chronologisch trennen, sodass kein Datenpunkt des Validierungsblocks in die Feature-Berechnung oder das Label des Trainingsblocks einfließt.
4. THE Feature_Pipeline SHALL einen automatisierten Kausalitätstest bereitstellen, der bei Ausführung bestätigt, dass kein Feature zum Zeitpunkt t Daten aus Zeitpunkten > t verwendet.

### Anforderung 7: Vorhersage-Logging und Validierung

**User Story:** Als Trader möchte ich, dass alle ML-Vorhersagen mit Zeitstempel, Features und tatsächlichem Ergebnis protokolliert werden, damit die Modellqualität retrospektiv überprüft werden kann.

#### Akzeptanzkriterien

1. WHEN der ML_Gate eine Vorhersage berechnet, THE Prediction_Logger SHALL folgende Daten in einer SQLite-Tabelle persistieren: Symbol, Vorhersagezeitstempel (UTC ISO-8601), vorhergesagtes Regime (trending oder choppy), Confidence_Score, Regime_Classifier-Wahrscheinlichkeit, LLM_Regime_Advisor-Konfidenz (falls vorhanden), die genutzten Top-Features mit ihren Werten, die resultierende Aktion (allow, block, hold) und den Risk_Scale-Faktor.
2. THE Prediction_Logger SHALL das tatsächliche Ergebnis (realisierte Performance der folgenden N Candles) nachträglich in derselben Tabelle ergänzen, sobald die Daten verfügbar sind.
3. THE Prediction_Logger SHALL eine Abfragefunktion bereitstellen, die für einen konfigurierbaren Zeitraum die Vorhersagegenauigkeit, die Rate blockierter Signale und den Vergleich zwischen vorhergesagtem und tatsächlichem Regime berechnet.
4. THE Prediction_Logger SHALL alle Vorhersage-Datensätze mit der Modellversion und dem Trainingszeitraum des verwendeten Regime_Classifier annotieren.

### Anforderung 8: Bestehende Sicherheits-Limits beibehalten

**User Story:** Als Trader möchte ich sicherstellen, dass der ML-Layer die bestehenden Risiko-Schranken des Bots nicht umgehen kann, damit das validierte Risikoprofil erhalten bleibt.

#### Akzeptanzkriterien

1. THE ML_Gate SHALL die bestehenden Stop-Loss-Berechnungen (ATR-basiert via `best_sl_tp`), Trailing-Stop-Logik (`next_atr_trailing_stop`) und Exposure-Limits (`exposure_headroom`) nicht verändern.
2. THE ML_Gate SHALL den Daily-Loss-Limit-Check (`_daily_loss_limit_hit`) nicht umgehen oder modifizieren.
3. THE ML_Gate SHALL die bestehende MAX_OPEN_POSITIONS-Begrenzung nicht umgehen.
4. THE ML_Gate SHALL ausschließlich den Entry-Zeitpunkt (Gate-Entscheidung) und die Positionsgröße (Risk_Scale) beeinflussen. Exit-Entscheidungen werden ausschließlich vom bestehenden Risk_Manager und der Trading_Loop gesteuert.
5. WHILE DRY_RUN auf "true" gesetzt ist, THE ML_Gate SHALL den DRY_RUN-Modus respektieren und keine realen Orders auslösen.

### Anforderung 9: Konfigurierbarkeit und Feature-Toggle

**User Story:** Als Trader möchte ich alle ML-Parameter über Umgebungsvariablen konfigurieren können, damit ich das ML-System ohne Codeänderungen aktivieren, deaktivieren und tunen kann.

#### Akzeptanzkriterien

1. THE ML_Gate SHALL über die Umgebungsvariable `ML_GATE_ENABLED` (Standardwert "false") aktiviert und deaktiviert werden.
2. THE ML_Gate SHALL die folgenden Umgebungsvariablen lesen: `ML_CONFIDENCE_THRESHOLD` (Standardwert 0.5), `ML_MIN_RISK_SCALE` (Standardwert 0.5), `ML_GB_WEIGHT` (Standardwert 0.7), `ML_LLM_WEIGHT` (Standardwert 0.3), `ML_LLM_ENABLED` (Standardwert "false"), `ML_RETRAIN_INTERVAL_DAYS` (Standardwert 30), `ML_MODEL_DIR` (Standardwert ".models/").
3. THE ML_Gate SHALL bei ungültigen Konfigurationswerten (z.B. negative Schwellenwerte, Gewichte, die nicht zu 1.0 summieren) einen ValueError mit beschreibender Fehlermeldung auslösen.
4. THE ML_Gate SHALL alle aktiven Konfigurationswerte beim Bot-Start im Log-Banner ausgeben, analog zur bestehenden `_banner`-Funktion.

### Anforderung 10: Exchange-Adapter-Abstraktion

**User Story:** Als Entwickler möchte ich, dass der Zugriff auf Marktdaten und Orderausführung hinter einer abstrakten Schnittstelle gekapselt ist, damit der Bot später auf Forex/MT5 oder andere Börsen erweitert werden kann, ohne die ML-Logik oder Kernstrategie zu ändern.

#### Akzeptanzkriterien

1. THE Exchange_Adapter SHALL eine abstrakte Basisklasse (oder ein Protocol) definieren, die folgende Methoden deklariert: `get_candles`, `get_current_price`, `place_order`, `close_order`, `get_account_info`, `get_top_symbols`.
2. THE Exchange_Adapter SHALL eine konkrete Implementierung für Bitget bereitstellen, die die bestehenden Funktionen aus bitget_client.py kapselt.
3. THE Exchange_Adapter SHALL so entworfen sein, dass eine zweite Implementierung (z.B. für MT5/Forex) hinzugefügt werden kann, ohne bestehende Adapter oder die ML_Gate-Logik zu ändern.
4. THE Exchange_Adapter SHALL die ausgewählte Implementierung über eine Umgebungsvariable `EXCHANGE_ADAPTER` (Standardwert "bitget") laden.

### Anforderung 11: Backtest-Integration der ML-Meta-Strategie

**User Story:** Als Trader möchte ich die ML-Meta-Strategie im bestehenden Portfolio-Backtest-Framework testen können, damit die Wirkung des ML-Gates auf historische Performance messbar ist.

#### Akzeptanzkriterien

1. THE Walk_Forward_Trainer SHALL eine Funktion bereitstellen, die für einen gegebenen historischen Zeitraum und ein gegebenes Symbol die Regime-Vorhersagen des trainierten Modells als Liste von (Zeitstempel, Regime, Confidence_Score)-Tupeln zurückgibt.
2. WHEN der Portfolio-Backtest mit aktiviertem ML-Gate ausgeführt wird, THE Portfolio-Backtest SHALL vor jeder simulierten Entry-Entscheidung die historische Regime-Vorhersage des ML-Modells abfragen und bei vorhergesagtem choppy-Regime den Entry blockieren.
3. WHEN der Portfolio-Backtest mit aktiviertem ML-Gate ausgeführt wird, THE Portfolio-Backtest SHALL den Risk_Scale basierend auf dem historischen Confidence_Score anwenden.
4. THE Portfolio-Backtest SHALL eine Vergleichsausgabe bereitstellen, die die Performance-Metriken (Sharpe-Ratio, Sortino-Ratio, maximaler Drawdown, Profit-Faktor, monatliche Renditen) mit und ohne ML-Gate gegenüberstellt.

### Anforderung 12: Modell-Persistenz und -Versionierung

**User Story:** Als Trader möchte ich, dass trainierte Modelle versioniert gespeichert werden, damit ein Rollback auf ein früheres Modell möglich ist und die Nachvollziehbarkeit gewährleistet bleibt.

#### Akzeptanzkriterien

1. THE Walk_Forward_Trainer SHALL jedes trainierte Modell zusammen mit einer Metadaten-Datei (JSON) speichern, die folgende Informationen enthält: Trainings- und Validierungszeitraum, Feature-Liste, Hyperparameter, Validierungsmetriken, Trainingszeitstempel und Daten-Hash.
2. THE Walk_Forward_Trainer SHALL die Modelldatei und die Metadaten im Verzeichnis `ML_MODEL_DIR` unter einem Unterverzeichnis mit dem Muster `{symbol}_{timestamp}/` ablegen.
3. THE ML_Gate SHALL beim Start das neueste verfügbare Modell je Symbol laden und den Pfad sowie die Metadaten im Log ausgeben.
4. IF mehrere Modellversionen für ein Symbol existieren, THEN THE ML_Gate SHALL eine Rollback-Funktion bereitstellen, die ein früheres Modell über einen Konfigurationsparameter aktiviert.

### Anforderung 13: Performance-Messung ohne Gewinnversprechen

**User Story:** Als Trader möchte ich, dass die Modellbewertung auf risikoadjustierten Metriken (Sharpe, Sortino) basiert und keine absoluten Renditeversprechen enthält, damit die Erwartungen realistisch bleiben.

#### Akzeptanzkriterien

1. THE Walk_Forward_Trainer SHALL die Modellqualität anhand von Sharpe-Ratio, Sortino-Ratio, maximalem Drawdown und Profit-Faktor auf dem Validierungsblock messen.
2. THE Prediction_Logger SHALL in Log-Ausgaben und gespeicherten Berichten keine absoluten Renditeversprechen oder Renditeprognosen formulieren.
3. THE ML_Gate SHALL beim Bot-Start den folgenden Hinweis im Log ausgeben: "ML-Gate ist ein experimentelles Filter-System. Vergangene Backtest-Ergebnisse garantieren keine zukünftige Performance."
4. THE Walk_Forward_Trainer SHALL empirisch gemessene Metriken ausschließlich mit dem Präfix "historisch gemessen" oder "Backtest-Ergebnis" kennzeichnen.

### Anforderung 14: Datenbank-Erweiterung für ML-Vorhersagen

**User Story:** Als Entwickler möchte ich, dass die bestehende SQLite-Datenbank um ML-spezifische Tabellen erweitert wird, ohne bestehende Tabellen zu verändern, damit die Datenmigration abwärtskompatibel bleibt.

#### Akzeptanzkriterien

1. THE Prediction_Logger SHALL eine neue Tabelle `ml_predictions` in der bestehenden SQLite-Datenbank anlegen, die ML-Vorhersagen mit den in Anforderung 7 definierten Feldern speichert.
2. THE Prediction_Logger SHALL eine neue Tabelle `ml_model_metadata` anlegen, die Metadaten trainierter Modelle referenziert.
3. THE Prediction_Logger SHALL die bestehenden Tabellen (trades, signals, funding_events, consumed_entry_signals, trade_quantity_epochs) nicht modifizieren.
4. THE Prediction_Logger SHALL die Tabellenanlage über die bestehende `init_db`-Funktion in database.py integrieren, wobei `CREATE TABLE IF NOT EXISTS` die idempotente Erstellung sicherstellt.
5. WHEN ein Trade über den ML_Gate gefiltert oder durchgelassen wird, THE Trading_Loop SHALL die `ml_prediction_id` als optionale Referenz in der signals-Tabelle speichern, ohne das bestehende Schema zu brechen.

### Anforderung 15: Fehlerresilienz des ML-Layers

**User Story:** Als Trader möchte ich, dass Fehler im ML-System den Bot nicht zum Absturz bringen, damit die bestehende Strategie auch bei ML-Ausfällen weiterarbeiten kann.

#### Akzeptanzkriterien

1. IF der Regime_Classifier bei der Inferenz eine Exception auslöst, THEN THE ML_Gate SHALL auf das Default-Verhalten (Confidence_Score=0.5, Regime=trending, Risk_Scale=1.0) zurückfallen und die Exception im Log protokollieren.
2. IF der LLM_Regime_Advisor bei einem API-Aufruf eine Exception auslöst oder ein Timeout überschreitet, THEN THE ML_Gate SHALL den LLM-Anteil ignorieren und ausschließlich den Regime_Classifier-Wert verwenden.
3. IF die Modell-Ladeoperation beim Bot-Start fehlschlägt, THEN THE ML_Gate SHALL im Fallback-Modus starten (gleichwertig zu `ML_GATE_ENABLED=false`) und eine Warnung protokollieren.
4. THE ML_Gate SHALL keine Exceptions an die Trading_Loop durchlassen, die den bestehenden Zyklus (Trailing-Stop-Refresh, Position-Management, Entry-Scan) unterbrechen.
