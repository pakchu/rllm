# EIA Weekly Natural Gas Storage Source Transport Rejection

## Decision

Reject the EIA Weekly Natural Gas Storage Report (`WNGSR`) as a historical
alpha source before opening any market, model, or portfolio outcome.

The report has a deterministic release clock and adequate nominal sample
count, but the official historical files do not preserve the exact regional
panel known at each weekly release. The source therefore cannot meet the
repository's historical/live causal-parity requirement.

## Official evidence

- Release schedule:
  <https://ir.eia.gov/ngs/schedule.html>
- Methodology:
  <https://ir.eia.gov/ngs/methodology.html>
- Revision policy:
  <https://ir.eia.gov/ngs/revisions.html>
- Published-file inventory:
  <https://ir.eia.gov/ngs/notice.html>
- Historical notes:
  <https://ir.eia.gov/ngs/notes.html>
- EIA automated-retrieval policy:
  <https://www.eia.gov/about/privacy_security_policy.php>

EIA normally releases WNGSR at 10:30 a.m. Eastern on Thursday and has
published the report since May 2002. That clock is usable.

The transport is not. The official namespace exposes live report files,
current history workbooks, and revision workbooks, but no date-coded
per-release HTML, PDF, CSV, JSON, or XLS archive. EIA describes the historical
database as updated weekly with the most current estimates, and its revision
policy permits regional and Lower-48 corrections. A later download can
therefore differ from the values a live policy saw.

## Prohibited repair

Do not:

- treat the mutable history workbook as a release-vintage ledger;
- reconstruct vintages from the current series and revision table;
- use FRED, ALFRED, Wayback, a vendor, or a search cache as a substitute;
- ignore regional revisions;
- claim exact historical/live parity from the fixed weekly clock alone; or
- open BTC outcomes to decide whether the source limitation is worth relaxing.

## Outcome boundary

No WNGSR production payload, event predicate, BTC row, funding row, return,
PnL, model prompt, adapter, or portfolio statistic was opened. This is a
transport rejection, not a negative alpha trial.
