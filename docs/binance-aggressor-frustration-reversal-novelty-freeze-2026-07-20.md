# BAFR-24F outcome-blind novelty freeze — 2026-07-20

## Decision

**PASS the preregistered clock-overlap gates.** BAFR-24F may advance to a
separately committed evaluator freeze. This is not a profitability or economic
independence claim. No BAFR market OHLC value, funding cash flow, post-entry
return, PnL, absolute return, CAGR, strict MDD, or hit rate was loaded.

The first implementation incorrectly attempted to reconstruct prior-family
clocks inside the BAFR novelty process and was rejected during review before it
was run. The repaired sequence is:

1. independently reconstruct and hash-freeze prior clocks without loading any
   BAFR source, clock, support result, or outcome;
2. commit that `signal_date`/`side`-only bundle; then
3. let the BAFR gate read only the frozen BAFR clock and frozen prior bundle.

## Frozen inputs

- BAFR support commit: `080e7ae`;
- BAFR clock file SHA256:
  `f3b816a76decce31136ed23d22f043eb8e80ef1b8697b869241b060062f01747`;
- prior-clock bundle commit: `ac814e7`;
- prior-clock bundle SHA256:
  `c5584256140799b380973f9f376e5751ad754a81c9683473467b9d05af0bb9f0`.

The bundle pins the prior result-artifact hashes, implementation-file hashes,
candidate parameters, clock counts, and canonical event-list hashes. MFIC and
NETF are gated by each frozen variant and by their timestamp-deduplicated family
union. WFRS and terminal absorption reproduce the original fit/select
reservation reset at `2023-01-01`.

## Frozen overlap result

The primary gate is a deterministic one-to-one timestamp match within ±12
five-minute bars. It deliberately ignores side for the primary match, which is
stricter for detecting a shared activation mechanism; same-side matches are a
diagnostic. Each comparison requires timestamp Jaccard `<= 0.20` and BAFR
containment `<= 0.30` on its declared common coverage.

| Comparator | BAFR / prior events | Time matches | Same-side | Jaccard | BAFR containment | Prior containment diagnostic | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| CBFR-72 | 3,027 / 144 | 115 | 60 | 0.0376 | 0.0380 | 0.7986 | pass |
| MFIC fast | 11,248 / 1,566 | 960 | 544 | 0.0810 | 0.0853 | 0.6130 | pass |
| MFIC slow | 11,248 / 1,635 | 943 | 563 | 0.0790 | 0.0838 | 0.5768 | pass |
| MFIC union | 11,248 / 3,019 | 1,452 | 897 | **0.1133** | **0.1291** | 0.4810 | pass |
| NETF fast | 11,248 / 319 | 240 | 88 | 0.0212 | 0.0213 | 0.7524 | pass |
| NETF slow | 11,248 / 267 | 200 | 60 | 0.0177 | 0.0178 | 0.7491 | pass |
| NETF union | 11,248 / 586 | 353 | 122 | 0.0307 | 0.0314 | 0.6024 | pass |
| WFRS L288 q90 H144 | 9,072 / 278 | 181 | 101 | 0.0197 | 0.0200 | 0.6511 | pass |
| terminal absorption W72 H72 | 10,173 / 100 | 61 | 26 | 0.0060 | 0.0060 | 0.6100 | pass |

The maximum frozen gate values are MFIC-union Jaccard **0.1133** and BAFR
containment **0.1291**, both below their limits.

## Important density caveat

BAFR is much denser than every comparator. Consequently, although the frozen
Jaccard and BAFR-containment gates pass, 48–80% of each sparse prior clock lies
within one hour of some BAFR event. Prior-clock containment was not a
preregistered rejection criterion and is therefore diagnostic only; it cannot
be promoted into a post-observation selection rule.

This weakens any broad claim of behavioral independence. The evaluator must
therefore preserve the original policy without threshold or holding-period
repair and must give special weight to transaction costs, the completed-bar
flow/rejection control, tick-component ablations, stale controls, and exact
side flip. A weak or control-replicated result retires BAFR rather than opening
a new search around this dense clock.

## Frozen output

- novelty result:
  `results/binance_aggressor_frustration_novelty_2026-07-20.json`
  (SHA256 `38ab5dbb1b36f14e32a4d7a09d94c37b84eaec5d1b75bbc5ef576660e05e3028`).

The next permitted action is to commit a strict evaluator and all required
controls before loading any BAFR post-entry price or funding outcome.
