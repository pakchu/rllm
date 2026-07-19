# PCBR-12 Premium Compression Breakout Relay preregistration

## Hypothesis

Two hours of unusually compressed premium-index motion followed by an efficient
ten-minute displacement that closes outside the compression cage and remains
pinned near its directional extreme can represent newly accepted derivative
demand. The frozen trade follows that premium displacement after one complete
empty five-minute latency bucket.

This is not a repair of PSR-30/6. PSR required high-energy alternating motion
that returned to its center and traded mean reversion. PCBR requires a quiet
context, efficient outside-cage displacement, terminal persistence, and trades
continuation. BTC price, volume, funding, OI, macro data, and existing alpha
state are absent from clock construction.

## Frozen rule

- Aggregate five exact valid one-minute premium-index rows into each completed
  five-minute premium OHLC bar.
- Context: 24 bars in `[T-130m,T-10m)`.
- Trigger: two bars in `[T-10m,T)`.
- Context range must be at or below its strictly-prior 30-day q25.
- Absolute trigger move must be at or above strictly-prior q90.
- Trigger efficiency must be at or above strictly-prior q70.
- Positive moves must close above the context high; negative moves must close
  below the context low.
- Directional terminal location must be at least `0.75`.
- Only a false-to-true onset signals. Side is the trigger-move sign.

Prior feature samples are shifted 26 five-minute bars, so their underlying
context and trigger paths cannot overlap the current 26-bar path. There is one
candidate, no threshold grid, no direction search, and no hold search.

## Execution

- Final source row is conservatively available at `T+1s`.
- Leave `[T,T+5m)` empty; enter BTCUSDT at `T+10m` open.
- Fixed one-hour hold, global non-overlap, 0.5x notional.
- Base cost 6 bp/notional/side; stress 10 bp/notional/side; exact funding.
- Full-calendar CAGR and strict intratrade MDD are mandatory.

## Sequential validation

Source-only support and novelty must pass before an evaluator is frozen. BTC
outcomes then open train (`2020-03` through 2022), test (2023), and eval
(`2024-01` through `2026-06`) in order, stopping at the first failure. Every
opened stage requires positive absolute and stress return, CAGR/strict MDD at
least 3, strict MDD at most 15%, weekly-cluster p at most 0.10, positive frozen
subperiods, and its minimum trade count. No failed result may change direction,
thresholds, latency, hold, sizing, or costs.
