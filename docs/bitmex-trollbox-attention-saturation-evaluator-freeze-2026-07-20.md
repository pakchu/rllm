# TBASR-24 evaluator freeze — 2026-07-20

The strict TBASR-24 evaluator was frozen after the evaluator source and its
outcome-blind preregistration were committed, and before any BTC OHLC or
funding row was parsed for this candidate.

## Frozen identities

- evaluator source SHA-256:
  `d32055317913bd80b00d0115bb0d5f26fa70b9f7d456d3718852e535a70ff193`;
- preregistration manifest:
  `0e9a7eb9cf61f23502fbe2779b4bd6e04c5f3718a4cdbcd2e3ac3fb1e698c42e`;
- semantic support result:
  `5996b7d7497d6bf5e96343f7ceca766363d58aa34280aea0fdb7b8653a8b1725`;
- semantic clock manifest:
  `fdcd9c7c376b18df2799acf24af04a421ca679e27009e6a539888defc7438aa8`;
- evaluator-freeze manifest:
  `3cde808815dc91283c65b23bb2462bf7d4cff087d6dff55c07f774dcd0707dc0`;
- evaluator-freeze file SHA-256:
  `36dde44985f26896fcc6ef861dc3a45c81479915038e5ea537cc2582f0b3b45a`.

## Outcome boundary at freeze

- train clear semantic events known before price: 1,728;
- test clear semantic events known before price: 990;
- parsed execution OHLC rows: 0;
- parsed funding rows: 0;
- execution data bytes hashed: false;
- price-conditioned schedules built: false;
- simulations run: false;
- opened windows: none;
- sealed windows: train, test;
- mutable parameters: none.

The freeze binds 24 checksum-frozen monthly market files permitted for the
train/warm-up prefix and 36 for the later test prefix. Runtime verification
recomputes those source contracts rather than trusting paths copied from the
freeze JSON. A test report can open only after the train report's complete
frozen gate evidence, no-search flag, no-repair flag, source hash, freeze hash,
and preregistration binding are all revalidated.

The next admissible action is the single train evaluation. Calendar 2022 must
remain unparsed unless every train gate passes.
