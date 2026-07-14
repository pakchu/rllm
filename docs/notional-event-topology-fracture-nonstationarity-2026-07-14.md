# NETF v1 nonstationarity diagnostic — 2026-07-14

## Scope and boundary

This is a post-hoc decomposition of the already opened 2020–2023 NETF v1
outcomes. It cannot repair, invert, promote, or retune NETF. The 2024, 2025,
and 2026 windows remain sealed.

- frozen rejection: `47cd225`
- frozen selection result SHA256: `478e6562ce59d6bfc93257d096381ccc19d5989a008771dfa9b797fb1334522a`
- diagnostic: `results/notional_event_topology_fracture_nonstationarity_2026-07-14.json`
- diagnostic SHA256: `4f3ea4e52fd36a33741a8bb4e5dcf0eeab93168b48a50d31b8ff77d540b6d52b`
- execution: next 5-minute open, scheduled-open exit, `0.5x`, `6 bp`
  notional cost per side
- CAGR: full calendar window including idle cash
- strict MDD: complete held path, favorable extreme before adverse extreme

## Calendar decomposition

### `netf_fast`

| year | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| 2020 | -6.92% | -6.91% | 10.22% | -0.68 | 140 |
| 2021 | +7.56% | +7.57% | 16.62% | 0.46 | 57 |
| 2022 | -4.37% | -4.38% | 8.36% | -0.52 | 49 |
| 2023 | -5.37% | -5.37% | 5.87% | -0.91 | 73 |

### `netf_slow`

| year | absolute return | CAGR | strict MDD | CAGR/MDD | trades |
|---|---:|---:|---:|---:|---:|
| 2020 | +9.88% | +9.86% | 5.92% | 1.67 | 111 |
| 2021 | +21.62% | +21.63% | 16.63% | 1.30 | 48 |
| 2022 | -6.65% | -6.65% | 11.43% | -0.58 | 46 |
| 2023 | -6.07% | -6.07% | 6.42% | -0.95 | 61 |

The slow candidate is not merely diluted by one bad selection year. Its sign
changes after 2021: both 2022 and 2023 lose, while 2020 and 2021 gain. The fast
candidate is even less stable, with only 2021 positive.

## Structure-combination decomposition

The three-bit code is ordered as:

1. arrival burst;
2. quote-notional concentration (HHI);
3. underlying trade-ID span per aggregate event.

The dominant slow-candidate state is `010`, meaning notional concentration
without an arrival-burst or trade-ID-span mark.

| year | combo | trades | mean account gross | mean account net |
|---|---:|---:|---:|---:|
| 2020 | `010` | 47 | +7.11 bp | +1.11 bp |
| 2021 | `010` | 19 | +35.44 bp | +29.41 bp |
| 2022 | `010` | 33 | -21.16 bp | -27.15 bp |
| 2023 | `010` | 56 | -6.15 bp | -12.14 bp |

The secondary `011` state is also unstable: `-1.71 bp` mean net in 2020,
`+51.13 bp` in 2021, and `+17.41 bp` in 2022, with no qualifying occurrence in
2023. Sparse combinations occasionally have large returns, but their sample
counts are too small to support a gate.

## Causal-state drift at signal origin

Median values at the slow candidate's setup origin changed materially:

| year | arrival burstiness | notional HHI | trades / aggregate event | relative topology tension |
|---|---:|---:|---:|---:|
| 2020 | 0.2611 | 0.0223 | 1.8088 | 1.3323 |
| 2021 | 0.1440 | 0.0084 | 2.1200 | 1.3599 |
| 2022 | 0.1141 | 0.0172 | 2.6007 | 1.2826 |
| 2023 | 0.0798 | 0.0260 | 2.6902 | 1.4268 |

Lagged ranks made the trigger causal, but they did not make its economic
meaning stationary. Aggregate-event composition, arrival shape, and venue
participation changed enough that the same rank-level topology state mapped to
opposite future returns.

## Root cause and successor constraint

NETF v1 encodes **levels and conjunctions**, not how relationships change
through time. Its frozen policy therefore cannot distinguish a 2021-style
concentration event from a superficially similar 2022–2023 event after the
market's event-generation process changed.

The next independent strategy must not be a post-hoc NETF gate. It should:

1. represent causal relations (`capital vs crowd vs price`) rather than raw
   magnitudes alone;
2. represent transitions between setup and confirmation (`rising`, `falling`,
   `recovered`, `diverged`);
3. learn whether to `abstain`, `follow`, or `fade` under a strictly historical
   state;
4. update only after each label horizon is complete, with purge/embargo before
   the next decision period;
5. keep 2024+ sealed until an independently preregistered 2020–2023 selection
   protocol is committed.

This motivates a single compact relational-token policy rather than another
fixed threshold conjunction or the previously rejected two-model
analyzer/trader architecture.
