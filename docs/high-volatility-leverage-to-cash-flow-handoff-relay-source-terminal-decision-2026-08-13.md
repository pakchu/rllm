# HVLCFH-8 terminal source-coverage rejection

The pushed HVLCFH-8 evaluator opened only native one-minute spot and perpetual
source rows. Perpetual fields were complete for all `3,747` decision blocks,
but the PostgreSQL spot OHLC history has NULL `quote_asset_volume` and
`taker_buy_quote` fields before the recent extension. Only 81 complete spot
flow blocks remained, all in July 2026; only three had the frozen ordered
leverage-to-cash handoff, so none could satisfy the 180-observation causal rank
warmup. Every split therefore had zero eligible events.

An immediate replay reproduced every artifact byte-for-byte:

- panel: `10a3671626db7e540a8a55f28241e907eab72555483f9163be449407be45b1fc`
- source manifest: `fc434e78e7b716ca9b46d54eee317e51f76990f7fd4992c78e77aa90a6f90326`
- primary clock: `3b4a342dc69628b913aff933bbb9411822c965b054f55a77e013961bf124e5e8`
- result: `345646b13580c27fb65634e7c21be6d7226e1355f3f2c1aef5b4bf39724fd6e4`

HVLCFH-8 is rejected unchanged before Gross9 or economics. Backfilling spot
flow, substituting price or volume, reducing warmup, or changing block, rank,
side, clock, hold, subset, threshold, comparator, or control is forbidden.
