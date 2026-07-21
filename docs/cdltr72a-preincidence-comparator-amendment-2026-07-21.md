# CDLTR-72A pre-incidence comparator amendment

## Disposition of CDLTR-72

`CDLTR-72` is rejected before preregistration and before any real component
value, feature incidence, event count, BTC market row, funding value, return,
PnL, CAGR, or MDD was opened.

Static review found that its novelty contract required signed occupied-exposure
correlation against every named prior clock even though two frozen artifacts do
not define a directional interval strategy:

- NWE-7 stopped at feature-support and has no frozen model side; and
- the 2023 live-anchor artifact freezes entries and sides but omits
  research-equivalent exits for its heterogeneous sleeves.

Inventing a side or hold after the fact would make the comparator arbitrary.
Silently exempting these clocks under the existing candidate identity would
weaken the committed mechanism. CDLTR-72 therefore stops without incidence.

## Successor identity

`CDLTR-72A` inherits the exact RRP, Cboe, network, relay, execution, windows,
support floors, controls, and LLM/RL boundary of CDLTR-72. Its only amendment is
a capability-aware comparator contract frozen before incidence.

### Timestamp novelty

Every named comparator must provide a nonempty sanitized decision clock and
must pass both gates:

```text
decision-date Jaccard <= 0.30
fraction of CDLTR dates within +/-1 UTC day <= 0.50
```

This applies to ORFR-1, CVTR-1, NTB-7, NWE-7, NWE-8, chain-activity impulse
momentum, FLCC-1, DFFB-601, the 2023 live-anchor clock, and every constituent
clock in the frozen prior-microstructure bundle.

### Signed occupied-exposure novelty

The absolute five-minute signed occupied-exposure Pearson cap of `0.40` applies
only when the frozen prior artifact already defines exact entry, exit, and side.
It is mandatory for ORFR-1, CVTR-1, NTB-7, NWE-8, chain-activity impulse
momentum, every frozen FLCC-1 candidate clock, and DFFB-601. The FLCC family
passes only when every candidate clock passes independently; no post-hoc union
or conflict resolver may be invented.

NWE-7, the 2023 live anchor, and the prior-microstructure bundle remain
timestamp-only. Their source artifacts do not provide an unambiguous complete
directional interval view. The successor may not invent, infer, or search a
missing side or exit. This limitation is explicit rather than treated as a
passing exposure result.

## Sanitized comparator boundary

Before CDLTR-72A preregistration, a dedicated normalizer must create one clean
clock bundle containing every named comparator and only:

```text
comparator, capability, decision_time, entry_time, exit_time, side, source_clock
```

For timestamp-only clocks, `exit_time` and `side` must be empty even when the
source contains a partial side hint. Their `source_clock` labels must also be
direction-neutral and may not retain LONG/SHORT tokens from source sleeve
names. Directional rows require finite side in `{-1,+1}` and
`decision_time <= entry_time < exit_time`.

Direct CSV clocks must be filtered by their already-frozen primary identity.
Outcome-bearing JSON artifacts may be read only through a streaming exact
top-level-subtree extractor that stops when the required clock subtree ends.
The chain-activity schedule must come only from the separately committed,
outcome-free comparator clock whose 2021, 2022, and 2023 schedule hashes match
the frozen pre-2024 manifest; the normalizer may not replay market, network,
funding, or execution code.

The normalizer may not compute or retain prior return, PnL, equity, CAGR, MDD,
or any CDLTR feature or event. Its output, manifest, implementation, inputs,
row counts, schemas, and complete comparator identity set must be hash-frozen
and committed before CDLTR-72A preregistration.

## No-repair boundary

This amendment is based only on static artifact capability, not observed CDLTR
incidence or performance. Once CDLTR-72A is preregistered, any source,
support, control, novelty, or later strict-economic failure rejects it without
changing comparator capability, thresholds, side, relay, latency, or hold.
