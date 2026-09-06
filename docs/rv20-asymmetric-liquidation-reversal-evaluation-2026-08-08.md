# RV20 Asymmetric Liquidation Reversal Evaluation

- Preregistration SHA-256: `832464ac5fee0d9585b3361d768b8a2ff40791e6f758501c466cbe6eade67e94`
- Verdict: **REJECT_NO_LEVERAGE**
- Selection: structural train top-1, then train leverage, then 2024 weight.
- Future use: veto only; no repair, rerank, rank-2, or threshold change.

## Frozen selection

```json
{
  "cell": {
    "max_hold_hours": 9,
    "rv20_regime_quantile": 0.75,
    "side_mode": "both",
    "stop_loss_pct": 0.04,
    "take_profit_pct": 0.04
  },
  "leverage": null,
  "weight": null
}
```

## Protocol integrity

- Structural cells tested: `216`
- Future used for ranking: `False`
- Repair attempted: `False`
- Rerank attempted: `False`
