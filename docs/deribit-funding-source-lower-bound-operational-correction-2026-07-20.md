# Deribit funding source lower-bound operational correction — 2026-07-20

## Trigger

After the DFIA-72 support preregistration was committed, the first complete
source attempt stopped in its first request window before writing any source
or manifest. The exact first hourly result, timestamp
`2019-04-30T10:00:00Z`, contained `prev_index_price=null`. The other four
fields were present and numeric. The next two bounded 28-day source windows
contained no nonnumeric value in any of the four numeric fields.

No complete source incidence, funding impulse, candidate count, Binance bar,
Binance funding row, future return, held path, or PnL was opened by the failed
attempt or the bounded diagnosis.

## Narrow correction

The loader will preserve this endpoint boundary fact without imputation:

- `prev_index_price=null` is accepted only for the exact configured first
  timestamp;
- it is serialized as an empty CSV field, counted explicitly in the source
  audit, and never converted to a price or return;
- any null at another timestamp remains a hard failure;
- the second and every later contiguous row must still link exactly to the
  preceding `index_price`; and
- an eight-hour feature/reference row remains ineligible until eight actual
  contiguous source rows are present.

The first row therefore contributes neither an index return nor a valid
eight-hour feature. It is retained only to bind the true source boundary and
subsequent price chain. No sign, threshold, reference window, latency, hold,
scheduler, support gate, or evaluation rule changes.

The original support artifact hash is
`911afce424443a7cd7e23b852357ce4bc1f6d0e837377b582515c8e88f6e7f41`.
A successor preregistration must disclose this source-only boundary diagnostic
and bind the corrected loader before another complete source attempt.
