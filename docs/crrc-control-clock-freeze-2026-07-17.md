# CRRC-72 mechanism-control clock freeze — 2026-07-17

Before any 2023 return was opened, five mechanism-removal clocks were frozen
with the same q85/q75/q55/q85 thresholds, `t+10m` entry, 72-bar hold, and
quarter-contained greedy scheduler as CRRC-72.

| Control | Events | Long | Short |
|---|---:|---:|---:|
| UM only | 714 | 382 | 332 |
| COIN-M only | 681 | 360 | 321 |
| credibility removed | 637 | 325 | 312 |
| inner-add only | 822 | 408 | 414 |
| outer-withdraw only | 1,067 | 545 | 522 |

These are **diagnostics, not repair candidates**. Their outcome cannot replace
or rerank the primary CRRC singleton. They test whether cross-venue agreement,
credibility, and radial add/withdraw conjunction add value over simpler and
much denser clocks.
