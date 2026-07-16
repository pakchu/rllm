# PFCR-1 preregistration — 2026-07-17

## Mechanism

At a common Binance USD-M funding settlement, PFCR reads the six rates only
after settlement. If the current cross-sectional spread exceeds the strictly
prior 180-event q90, it buys the lowest-funding alt and shorts the highest-
funding alt. Causal 30-day betas set gross-one factor-neutral weights.

The signal becomes available at settlement +5 minutes, enters at +10 minutes,
and exits four hours later. The just-observed settlement belongs to neither
leg because entry occurs afterward. This tests leveraged crowding release, not
funding carry.

## Qualification

Support is checked without post-entry returns. The singleton must then be
positive in both 2023 and 2024, achieve each-year ratio >=1.5, combined full-
calendar CAGR/strict-MDD >=3, strict MDD <=15%, at least 60 trades, positive
10 bp and +5m-delay controls, inferior direction flip, and weekly cluster
p<=0.10. Only a pass can open 2025; only a 2025 pass can open 2026.

## Boundary

The six-alt rows were used by adjacent research, so historical performance is
not pristine enough for live promotion. Even a full historical pass requires
atomic two-leg execution parity and at least 90 forward-shadow days. No sign,
threshold, hold, beta, or pair repair is allowed after an outcome window opens.

Protocol hash: `bc68b2788d1a67f28fad7744ff480af58da0a111a2fcd48cc42b4288d4e57528`
