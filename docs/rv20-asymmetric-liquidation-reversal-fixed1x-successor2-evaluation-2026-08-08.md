# RV20 Asymmetric Liquidation Reversal Fixed1x Evaluation

- Preregistration SHA-256: `fb7b536cf4ae5d1dcc98fd095118ac84206dc89035d698cc13a5bec640ec2854`
- Verdict: **REJECT_NO_2024_WEIGHT**
- Selection: exact train replay, then 2024 Gross9 weight only.
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
