# RV20 Low-Participation Shock Reversal Breakeven1 Fixed1x Evaluation

- Preregistration SHA-256: `f2738dca6a29c697aea9a05a45cd4b4a42d1ebef8476c02fa100f63424ab68ae`
- Verdict: **REJECT_NO_2024_WEIGHT**
- Selection: exact train replay, then 2024 Gross9 weight only.
- Future use: veto only; no repair, rerank, rank-2, or threshold change.

## Frozen selection

```json
{
  "cell": {
    "max_hold_hours": 12,
    "rv20_regime_quantile": 0.6,
    "side_mode": "both",
    "stop_loss_pct": 0.04,
    "take_profit_pct": 0.04
  },
  "leverage": 1.0,
  "weight": null
}
```

## Protocol integrity

- Structural cells tested: `1`
- Fixed leverage: `1.0`
- Future used for ranking: `False`
- Repair attempted: `False`
- Rerank attempted: `False`
