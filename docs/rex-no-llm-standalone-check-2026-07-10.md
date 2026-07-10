# REX no-LLM standalone check (2026-07-10)

This report disables LLM inference and evaluates REX in two non-LLM forms: raw source candidates and the fixed dual-regime rule gate.

- Result JSON: `results/rex_no_llm_standalone_check_2026-07-10.json`
- No model adapter / LoRA / LLM selector used.
- CAGR denominator is the full split window including idle time.
- Strict MDD includes intraposition adverse excursion.

## source_all

events: 701

- train: ret 34.05% | CAGR 9.19% | strict MDD 39.91% | ratio 0.23 | trades 486 | win 49.6% | sides {'LONG': 268, 'SHORT': 218}
- test2024: ret 3.43% | CAGR 3.42% | strict MDD 15.92% | ratio 0.22 | trades 105 | win 49.5% | sides {'LONG': 82, 'SHORT': 23}
- eval2025: ret 19.48% | CAGR 19.49% | strict MDD 9.95% | ratio 1.96 | trades 70 | win 52.9% | sides {'LONG': 29, 'SHORT': 41}
- ytd2026: ret 4.38% | CAGR 10.79% | strict MDD 9.14% | ratio 1.18 | trades 40 | win 55.0% | sides {'SHORT': 26, 'LONG': 14}

## rule_gate

events: 475

- train: ret 89.39% | CAGR 21.13% | strict MDD 28.08% | ratio 0.75 | trades 356 | win 53.1% | sides {'LONG': 173, 'SHORT': 183}
- test2024: ret 38.79% | CAGR 38.70% | strict MDD 11.28% | ratio 3.43 | trades 62 | win 56.5% | sides {'LONG': 48, 'SHORT': 14}
- eval2025: ret 38.77% | CAGR 38.80% | strict MDD 5.12% | ratio 7.57 | trades 33 | win 69.7% | sides {'LONG': 7, 'SHORT': 26}
- ytd2026: ret 11.02% | CAGR 28.33% | strict MDD 7.37% | ratio 3.84 | trades 24 | win 62.5% | sides {'SHORT': 23, 'LONG': 1}

## rule_gate_short_only

events: 246

- train: ret 38.78% | CAGR 10.33% | strict MDD 23.80% | ratio 0.43 | trades 183 | win 52.5% | sides {'SHORT': 183}
- test2024: ret -0.43% | CAGR -0.43% | strict MDD 6.98% | ratio -0.06 | trades 14 | win 42.9% | sides {'SHORT': 14}
- eval2025: ret 29.94% | CAGR 29.97% | strict MDD 5.12% | ratio 5.85 | trades 26 | win 69.2% | sides {'SHORT': 26}
- ytd2026: ret 10.11% | CAGR 25.85% | strict MDD 7.37% | ratio 3.51 | trades 23 | win 60.9% | sides {'SHORT': 23}
