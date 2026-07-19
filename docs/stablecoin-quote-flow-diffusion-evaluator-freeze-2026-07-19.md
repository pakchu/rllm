# SQFD-6 strict evaluator freeze — 2026-07-19

SQFD-6 is now frozen for one-way sequential outcome evaluation. This artifact
does **not** claim profitability. It locks the only evaluator permitted to open
train, then test, eval, and final in that order, stopping permanently at the
first failed preregistered gate.

## Frozen identity

- evaluator: `training/evaluate_stablecoin_quote_flow_diffusion.py`
- evaluator SHA-256:
  `0ea59a107f05777ba91ab1c8fc5900e724ba48ec6ce647a42c34c34422222e3b`
- freeze artifact:
  `results/stablecoin_quote_flow_diffusion_evaluator_freeze_2026-07-19.json`
- freeze artifact SHA-256:
  `84b318982d58b4805805e9907970c95b9cc2fb1b7c330f263b90ee6f3b4bc47c`
- freeze manifest:
  `f7913f23ef090658e11641e50f83581e8d3066318f74ac12c21c9e0da7d36903`
- frozen source-only support commit:
  `107ddbddc233e9316392cc15ba69588f497e06db`

## Outcome isolation at freeze

- opened windows: `[]`
- sealed windows: `train`, `test`, `eval`, `final`
- execution OHLC rows parsed: `0`
- funding rows parsed: `0`
- raw execution data bytes hashed: `false`
- simulations run: `false`
- mutable parameters: `[]`

The freeze reads only checksum-locked source-only clocks, preregistration,
support artifacts, and execution-source manifests. The raw train OHLC and
funding bytes remain unopened until the committed evaluator executes train.
Direct future-window loaders also verify every prior passing report before
opening a future source manifest or physical data file.

## Accounting and gate contract

- fixed `0.5x` notional;
- `6 bp/notional/side` base cost and `10 bp/notional/side` stress cost;
- exact six-hour / 72-bar exit with no stop, take-profit, or dynamic exit;
- exact funding debits retained at entry/exit boundaries, boundary credits
  dropped, and every settlement mark still visited for strict MDD;
- global/pre-entry high-water mark, entry fee, every held 5m
  favorable-then-adverse OHLC path, virtual adverse-mark exit cost, and actual
  exit cost;
- full declared calendar CAGR, including all idle seconds;
- deterministic UTC ISO-week cluster sign-flip test;
- primary margin checked against all seven frozen controls, including the
  direction flip, one-hour latency, and deterministic random-side controls;
- exact zero-MDD ratio semantics, with no epsilon substitution.

Positions own `[entry_time, exit_time)`. Therefore an exit exactly at a split
end is contained by the frozen support contract. A split-end market open or
funding event is parsed only when a frozen schedule actually requires that
boundary. None of the current SQFD schedules requires one.

Funding boundaries use Binance's retained **exact `fundingTime`**, not the
nominal eight-hour grid label. Thus an event stamped 47 ms after entry is an
interior event and is settled symmetrically; an event stamped 47 ms after exit
is outside the position. Only an event whose exact timestamp equals entry or
exit receives the conservative boundary-credit treatment.

The 2020–2023 funding source preserves exact Binance funding times and rates.
Where Binance's historical funding endpoint did not return a settlement mark,
the already frozen source uses the official containing 8h mark-price kline open
proxy. Its source manifest reports a maximum implied funding-cash error of
`0.00134844 bp/notional`; this caveat remains explicit in every downstream
interpretation.

## Verification evidence

- SQFD preregistration, support, evaluator, and artifact suite: `51 passed`;
- evaluator-focused suite after final fixes: `31 passed`;
- Ruff: clean;
- Pyright: `0 errors, 0 warnings`;
- two independent temporary freezes: byte-identical;
- adversarial code re-review: `APPROVE`, zero remaining findings.

The only next permitted action is opening **train 2023-H2** once. A failed train
gate rejects SQFD-6 without repair and leaves test/eval/final sealed.
