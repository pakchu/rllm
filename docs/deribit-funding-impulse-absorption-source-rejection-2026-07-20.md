# DFIA-72 — outcome-blind source rejection

## Decision

**Reject DFIA-72 before candidate construction or market evaluation.** The
frozen Deribit Funding-Impulse Absorption singleton failed its precommitted
one-hour/eight-hour source-memory invariant. The threshold will not be widened
after seeing the complete source prefix.

No Binance bar, Binance funding row, future return, held path, PnL, candidate
clock, absolute return, CAGR, or strict MDD was loaded or calculated. No source
gzip, source manifest, support result, or event clock was written.

## Frozen evidence chain

- mechanism/source decision: commit `6405323`;
- initial source loader: commit `0126e85`;
- original support freeze: commit `72bcf46`, artifact hash
  `911afce424443a7cd7e23b852357ce4bc1f6d0e837377b582515c8e88f6e7f41`;
- lower-bound null correction: commit `d11c557`;
- successor support freeze: commit `174c8e1`, artifact hash
  `e974ceba3e824e5da078a97ef396080261b020bee0999e34a75af8ce3c908a03`;
- corrected loader SHA-256:
  `ef166913bf398282056a046e15985e2b8f2a81d8f338376fcf5ee2f8cc21d00d`;
- successor preregistration source SHA-256:
  `a9dd7ab37235bb567ba6d0654d4ae0d9437def8861b37cd80d6b0d136a53196e`;
  and
- successor preregistration file SHA-256:
  `126677f5904819d8ac77ef5a6e091c5b1a9ba452a7a623c249986168ecd3f952`.

## Exact frozen failure

The corrected loader fetched the frozen interval
`[2019-04-30T10:00:00Z, 2024-01-01T00:00:00Z)` into memory and then ran the
precommitted source audit before any file write. It stopped at:

```text
timestamp = 2019-05-26T22:00:00Z
abs(sum(latest 8 interest_1h) - interest_8h)
          = 0.000068554146824142269
frozen maximum = 0.00005
```

The observed error is about `1.3711×` the frozen maximum. The invariant was an
explicit source-quality requirement in the decision, loader, and successor
preregistration. Relaxing it now, changing the memory algebra, excluding early
history, or selecting a favorable subperiod would be post-source repair.

## Opened and sealed boundaries

The complete Deribit pre-2024 response prefix was fetched transiently, so its
source incidence is disclosed as opened. Raw responses were not persisted and
the loader failed before deterministic source/manifest writes. The funding
impulse, causal references, candidate incidence, support counts, and every
Binance outcome remain unopened.

Therefore:

- no DFIA event clock exists;
- no strict evaluator is authorized;
- no performance statistic can be reported; and
- DFIA-72 may not be revived by changing its source tolerance, threshold,
  side, reference, latency, hold, scheduler, or support gate.

The next BTC candidate must use a genuinely different observable/mechanism.
