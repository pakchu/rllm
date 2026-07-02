# Event candidate family state cards with position state (2026-07-02)

## Answer to the position-state question

Before this pass, the new family-scoreboard input did **not** include current position state.  That is unacceptable for live trading because the model must know whether it is flat, long, short, already in profit/loss, or carrying risk before selecting a new family/action.

This pass adds an explicit `position_state` block to every LLM state-card record.

Historical fold records use:

```json
{
  "mode": "FLAT",
  "side": "NONE",
  "size_pct": 0.0,
  "entry_price": null,
  "entry_time": null,
  "age_bars": 0,
  "unrealized_pnl_pct": 0.0,
  "source": "historical_fold_default; live runner must overwrite from exchange/wave_trading"
}
```

In live/testnet execution this field must be overwritten from Binance/wave_trading position state before inference.

## Implementation

New script: `training/build_event_candidate_family_state_cards.py`

Inputs:

- selector report with `pre_fold_scoreboard`;
- max options per prompt;
- optional `ABSTAIN` option;
- default historical position mode.

Output per JSONL row:

- `fold`
- `position_state`
- `options`
- `target`
- `prompt`
- `completion`
- leakage guard metadata

## Generated PoC data

Command:

```bash
.venv/bin/python -m training.build_event_candidate_family_state_cards \
  --selector-report results/event_candidate_regime_family_selector_rex_core_abstain_scoreboard_6m_2023_2026h1_2026-07-02.json \
  --output-jsonl data/event_candidate_family_state_cards_rex_core_6m_2023_2026h1_2026-07-02.jsonl \
  --max-options 5 --split-name rex_core_2023_2026h1
```

Summary:

- rows: 7
- targets: `A=6`, `ABSTAIN=1`
- prompt includes `Current position: {...}` before options.

This is a schema PoC, not enough data for meaningful SFT.  The next data expansion should use monthly folds or per-signal state cards.

## Verification

- `py_compile` passed for exporter and tests.
- Manual tests passed:
  - state cards include explicit FLAT position state;
  - prompt includes `Current position:`;
  - JSONL writer works.

## Next action

Generate a larger state-card dataset:

1. monthly or 2-month folds instead of 6-month folds;
2. live-compatible position fields populated from wave_trading/Binance testnet adapter;
3. listwise Gemma prompt over family options plus `ABSTAIN`;
4. evaluate with chronological train/test/eval split before any live use.
