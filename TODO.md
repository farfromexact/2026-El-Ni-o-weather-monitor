# Current Goal
- Optimize the weather-driven sugar and palm oil long-futures monitoring app until it is decision-useful for a professional trader: transparent data quality, robust signal logic, explicit trade triggers, risk controls, and verifiable daily operation.

# Done
- Built a Streamlit monitor for sugar SR and palm oil P with weather risk, price confirmation, historical templates, iFinD defaults, and near-month dual-commodity comparison.
- Fixed iFinD default symbols to `SRZL.CZC` and `PZL.DCE`.
- Verified `python -m py_compile app.py`, live iFinD data retrieval, Open-Meteo weather retrieval, and browser rendering on `http://localhost:8501/`.
- Added a signal health/data quality gate that shows weather coverage, price sample length, price freshness, and data warnings before trade interpretation.
- Verified the health gate with `python -m py_compile app.py`, live iFinD/Open-Meteo function checks, and browser rendering on `http://localhost:8501/`.
- Added a separate Regime background layer using NOAA CPC ONI and commodity-specific seasonal production windows; it is displayed outside the short-term weather score.
- Verified Regime background with `python -m py_compile app.py`, live NOAA ONI parsing, function-level regime checks, and browser rendering of the `Regime背景` tab.
- Added position sizing and risk budget outputs based on ATR stop distance, contract multiplier, account equity, risk budget, margin cap, trial fraction, and estimated margin rate.
- Verified position sizing with `python -m py_compile app.py`, live iFinD calculation checks, and browser rendering of the `仓位与风险预算` section.
- Added a historical case replay worksheet with preset narrative trigger dates, iFinD historical price loading, market-confirmation detection, lag, max favorable/adverse move, stop-hit status, and replay charts.
- Verified case replay with `python -m py_compile app.py`, live iFinD historical case checks for sugar/palm, and browser rendering of the `历史案例价格路径回放` section.
- Added a daily run log/export layer with current signal snapshots, explicit write action, local CSV persistence at `logs/signal_run_log.csv`, recent-record display, and CSV downloads.
- Verified the run log with `python -m py_compile app.py`, live signal snapshot file-write/read checks, and browser rendering of the `运行日志` tab.
- Added an alert/change summary that compares the current snapshot with the previous same-commodity run and highlights action/gate changes, material score moves, threshold crossings, and key price/risk deltas.
- Verified the change summary with `python -m py_compile app.py`, function-level no-change and threshold-crossing checks, and browser rendering of `变化提醒` in the `运行日志` tab.
- Added an all-commodity morning summary that computes both sugar and palm oil signal snapshots from the same weather, price, health, regime, and risk logic, then renders them side by side before the selected-detail view.
- Verified the morning summary with `python -m py_compile app.py`, synthetic two-commodity snapshot/table checks, and browser rendering of `盘前双品种总览` with data health, Regime, price confirmation, and risk outputs for both commodities.
- Added weather anomaly persistence and percentile diagnostics: historical percentile ranks for forecast precipitation, temperature, and water balance; recent same-window actual stress; persistence labels/multipliers; and raw versus adjusted weather stress scores.
- Verified weather persistence diagnostics with `python -m py_compile app.py`, function-level persistence/summary checks, and browser rendering of the new persistence, precipitation-percentile, and recent-rain-ratio fields on `http://localhost:8501/`.
- Added a contract liquidity and roll-risk gate that validates commodity-symbol fit, latest trade date, volume/open-interest quality, zero-volume days, calendar gaps, price-jump continuity, observed contract-code changes, and delivery-month roll proximity before trade guidance is allowed.
- Wired the contract gate into signal health, action gating, position output, morning summary, selected-detail signal quality UI, and run-log export fields.
- Verified contract checks with `python -m py_compile app.py`, synthetic pass/block/caution unit checks, summary-table checks, and browser rendering of contract status, liquidity, and continuity diagnostics on `http://localhost:8501/`.
- Added an active delivery-month contract selector for iFinD data: it generates 1/5/9 candidate months, formats exchange-specific symbols, fetches candidate histories through the same iFinD HTTP/SDK path, ranks candidates by open interest and volume, and falls back to volume-only selection when iFinD HTTP does not provide open interest.
- Wired resolved contract symbols and selector metadata through signal health, contract checks, morning summary, selected-detail UI, and run-log export so the app records which tradable contract replaced the continuous symbol and why.
- Verified the active-contract selector with `python -m py_compile app.py`, synthetic full-OI and volume-only candidate tests, mocked iFinD `get_price_data` integration, live iFinD selection of `SR609.CZC` and `P2609.DCE`, and browser rendering of candidate-selection diagnostics on `http://localhost:8501/`.
- Added commodity-specific weather impact timing: each regional weather stress is classified as immediate, 1-3 month, or 3-9 month impact, then translated into a time-lag-adjusted entry weather score while preserving the original weather-pressure score for audit.
- Wired impact timing into regional weather scoring, commodity signal snapshots, top-driver ranking, morning summary, weather map hover data, selected-weather detail table, trigger-condition display, and run-log export fields.
- Verified impact timing with `python -m py_compile app.py`, synthetic sugar/palm timing checks, synthetic summary/log checks, and browser rendering of `盘前双品种总览`, `入场天气`, and `影响时滞` on `http://localhost:8501/`.
- Added a scenario-based entry trigger playbook that separates watch, price trigger, trial entry, add-on, and invalidate conditions for each commodity using the time-lag-adjusted weather score, price confirmation, combined score, contract/data gate, and risk plan.
- Wired the playbook into commodity snapshots, morning summary, selected-detail execution framework, current-run log export, and change alerts so daily records capture the active execution stage and next trigger.
- Verified the playbook with `python -m py_compile app.py`, synthetic watch/price-trigger/trial/add-on/invalidate stage checks, synthetic summary/log checks, and browser rendering of `执行阶段`, `下一触发`, and `情景化入场 Playbook` on `http://localhost:8501/`.
- Added a post-entry management playbook for each commodity with trailing stop, time stop, partial exit, no-add, and de-risk rules after a trial entry, using commodity-specific holding windows and ATR/R-multiple logic.
- Wired post-entry management into commodity snapshots, morning summary, selected-detail execution framework, current-run log export, and change alerts so daily records capture management status, trailing stop, partial-exit reference, time-stop window, and current R multiple.
- Verified post-entry management with `python -m py_compile app.py`, synthetic pre-entry/no-add/trailing-stop/partial-exit/de-risk checks, synthetic summary/log checks, and browser rendering of `试仓后管理 Playbook`, `移动止损`, `时间止损`, `部分止盈`, `不加仓`, and `降风险` on `http://localhost:8501/`.

- Added manual position-state inputs and CSV persistence at `logs/position_state.csv` for actual trial-entry date, average entry price, lots, and notes.
- Wired persisted position state into dual-commodity snapshots, post-entry management, morning summary, run-log export, and time-stop logic so actual holding days can automatically trigger `时间止损`.
- Verified position-state persistence and time-stop wiring with `python -m py_compile app.py`, synthetic save/load/holding-days/log/summary checks, and browser rendering of `实际持仓状态`, `保存持仓状态`, `持仓管理`, and `试仓后管理 Playbook` on `http://localhost:8501/`.

- Added a portfolio-level exposure and stacking gate with sidebar controls for portfolio risk cap, portfolio margin cap, same-direction correlation threshold, and correlation lookback window.
- Wired the portfolio gate into dual-commodity snapshots, entry playbook downgrades, morning summary, selected execution framework, run-log export fields, and change alerts so sugar and palm oil cannot both scale up when risk, margin, or correlation constraints are breached.
- Verified the portfolio gate with `python -m py_compile app.py`, synthetic high-correlation and limited-capacity checks, signal-log/table assertions, and Streamlit `AppTest` rendering of `组合风险/叠加闸门`, `组合风险闸门`, and the new sidebar controls.

- Added last-good-data caching under `logs/cache` for weather tables, NOAA ONI/ENSO data, and commodity price histories, with CSV data files and JSON metadata for saved time, key, and row count.
- Wired stale fallback into weather, ENSO, AKShare, and iFinD price paths so API outages can reuse the last successful pull while adding explicit last-good warnings, source-status UI, signal-health degradation, and run-log source fields.
- Verified last-good fallback with `python -m py_compile app.py`, synthetic weather/ENSO/price outage checks, health-gate warning assertions, and Streamlit `AppTest` rendering of the data-source status message.

- Added a daily pre-trade checklist and operator sign-off layer before run-log writes, with automatic checks for data status, contract status, portfolio gate, planned lots, and invalidation level.
- Wired the checklist into the current snapshot/export fields and disabled actionable trade-day logging until required system checks pass and an operator name plus manual approval are recorded.
- Verified the checklist with `python -m py_compile app.py`, function-level actionable/blocking/non-actionable checks, and Streamlit `AppTest` rendering of `盘前交易 Checklist / 人工确认`.

- Added a post-trade outcome tracker that scans signed actionable log entries and links each signal to the latest available price path for MFE/MAE, latest R, stop-hit, partial-profit, and time-stop outcome classification.
- Added persistent realized execution notes at `logs/post_trade_notes.csv`, including actual execution status, realized average price, realized lots, review notes, and update time, plus a CSV export for the outcome table.
- Verified the outcome tracker with `python -m py_compile app.py`, synthetic signed-signal price-path tests for partial-profit and stop outcomes, note persistence assertions, and Streamlit `AppTest` rendering of `Post-trade Outcome Tracker`.

- Added model-calibration diagnostics that merge signed actionable logs with outcome records, then summarize realized MFE/MAE, latest R, hit rate, stop rate, partial-profit rate, and time-stop rate by commodity, weather driver, entry stage, data quality, and portfolio gate.
- Wired the diagnostics into the post-trade outcome tracker with sample-size-aware recommendations, detail samples, and a calibration CSV export; the panel now renders even when samples are still insufficient.
- Verified calibration diagnostics with `python -m py_compile app.py`, synthetic signed-outcome grouping checks, recommendation assertions, and Streamlit `AppTest` rendering of `模型校准诊断`.

- Added a threshold-review workbench that compares current weather, price, and combined build thresholds against signed historical outcomes using candidate threshold grids around the live settings.
- Wired the workbench into calibration diagnostics with current-threshold baseline metrics, conservative candidate recommendations, CSV export, and an explicit guarantee that suggestions do not auto-change live sidebar rules.
- Verified threshold review with `python -m py_compile app.py`, synthetic signed-outcome threshold-grid checks, current-baseline assertions, and Streamlit `AppTest` rendering of `阈值审阅 Workbench`.

- Added a regime-aware stress-test panel that simulates weather-signal reversal, -1ATR and -2ATR overnight gaps, direct stop breaches, and 5% adverse price gaps against actual/open positions or approved planned lots.
- Wired stress tests into the build-signal tab with estimated loss, account loss percentage, stressed margin usage, regime/weather context, CSV export, and forced action recommendations such as stop out, cancel add-on, halve trial position, or no-add review.
- Verified stress testing with `python -m py_compile app.py`, synthetic actual-position and approved-planned-lot scenarios, forced-action assertions, and Streamlit `AppTest` rendering of stress-test warnings.

- Added an execution calendar and market-session guard that classifies pre-open, intraday, closing-window, close-confirmed, weekend/holiday, and stale-session states from China time plus the latest price date.
- Wired the session guard into commodity snapshots, action wording, portfolio-gate overrides, morning summary, selected execution framework, run-log export fields, and the pre-trade checklist so actionable language is only allowed after close-confirmed data.
- Verified session gating with `python -m py_compile app.py`, synthetic session/action/checklist tests, and Streamlit `AppTest` rendering of the trading-session UI.
# Next
1. Add a notification-ready alert export layer that produces concise daily trade-desk messages for watch, trigger, block, and forced-action states, including data/session caveats and required operator actions.

# Blockers
- None.