# PFCR-2 episode-onset preregistration — 2026-07-17

## Mechanism

PFCR-2 retains PFCR-1's causal post-settlement, six-alt, beta-neutral pair.
It accepts only the first eligible settlement at least **36 hours** after the
previous accepted settlement. This treats repeated extreme settlements as one
crowding episode rather than independent signals.

## Outcome-blind derivation

PFCR-1 was rejected before any post-entry return was calculated because its
maximum monthly event share was 24.29%, above the frozen 20% support limit.
Only timestamps, selected symbols, and support concentration were inspected.
Weekly first/maximum rules and 36/48/60/66/72/84/96-hour cooldowns were checked;
36 hours was the shortest inspected deterministic cooldown satisfying every
unchanged support gate. PFCR-2 is therefore a separately frozen protocol, not
an outcome-based repair of PFCR-1.

## Qualification

The support and return gates remain unchanged: at least 60 events and 25 per
year, broad pairs/symbols, maximum pair share 25%, maximum month share 20%,
then positive 2023 and 2024 returns, each-year CAGR/strict-MDD >=1.5,
combined ratio >=3, strict MDD <=15%, at least 60 trades, and all frozen
robustness controls. Only a pass can open 2025, then 2026.

Protocol hash: `7dc7d51af83f4d4ce9822439799277d39ddd87a181b8728b468beedc5080d3d1`
