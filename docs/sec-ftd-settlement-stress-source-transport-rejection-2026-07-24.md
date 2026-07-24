# SEC FTD settlement-stress source transport rejection — 2026-07-24

## Decision

Retire:

```text
SFTD-v1 — SEC/NSCC Fails-to-Deliver Settlement-Stress Ledger
```

at the production-index transport boundary.

The frozen direct collector cannot read the sole official archive index from
this runtime. No source builder, archive download, mechanism, candidate,
market join, model, or economic evaluation is authorized.

This is terminal. SFTD-v1 may not be retried, resumed, repaired, narrowed,
mirrored, proxied, or relabeled.

## Bound source identity

The independently reviewed source boundary was committed as:

- commit: `6ac40914002955ef7d5323ae75636cfee0d97e53`;
- document:
  `docs/sec-ftd-settlement-stress-source-axis-decision-2026-07-24.md`;
- SHA-256:
  `475e3616848a3e6c8914a8ed55ed71d99efdbec1b678e402f8633f565fd6cdc4`.

The sole frozen index was:

<https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data>

## Direct transport evidence

After the boundary commit, one metadata/index-only preflight used:

- direct TLS through Python `http.client`;
- host `www.sec.gov`;
- method `GET`;
- the exact frozen index path;
- a contact-bearing research `User-Agent`;
- `Accept-Encoding: identity`;
- no proxy, cookie, browser session, API key, authorization, or alternate
  endpoint; and
- no redirect.

The response was:

```text
status: 403
content-type: text/html
content-length: 1925
bytes read: 1925
```

The response body was an access-denied HTML document, not an FTD index or
archive. No archive URL was requested and no FTD ZIP byte, text row, symbol,
CUSIP, quantity, price, incidence, candidate, market value, or outcome was
opened.

An earlier pre-boundary direct `curl` request to the same index with a
contact-bearing research user agent also returned HTTP 403. That earlier
failure was not used as a substitute source or transport.

## Why search-engine visibility is irrelevant

Official SEC documentation and search tools can render the public landing
page, but a search/browser retrieval service is not the production collector
frozen by SFTD-v1. Depending on such a service would add an undeclared
third-party transport, browser identity, cache, and availability clock.

The boundary expressly forbids:

- cookies or browser sessions;
- proxies;
- alternate SEC paths or CDN rewrites;
- search-engine caches;
- mirrors; and
- user-agent tuning after source transport is tested.

Using any of those to bypass the direct HTTP 403 would make historical
research and live collection different systems. It is therefore prohibited.

## Why the builder is not implemented

The next planned unit was a synthetic-tested source builder followed by a
single archive audit. The exact production index is a prerequisite for
discovering the 108 frozen archive links. Implementing the builder after its
only admissible index transport has failed would create dead code and invite a
later transport repair.

Stopping before archive access is the strongest no-repair outcome:

- zero source rows were exposed;
- zero candidate decisions were made;
- zero outcome values were opened; and
- no one-shot archive sentinel was consumed.

The terminal rejection is bound in:

```text
results/sec_ftd_settlement_stress_source_transport_rejection_2026-07-24.json
```

## Consequence

SFTD-v1 contributes no alpha, feature, gate, model input, portfolio weight, or
live dependency.

The next alpha search must select a different official source whose direct,
keyless or contractually authorized collector is proven accessible before a
source boundary commits to a full builder.
