# HVSOF-8 source-support result

The six frozen action-by-state candidates were materialized twice without opening
prices, returns, funding, PnL, or Gross9 comparator rows. Twelve preregistration
and support tests passed twice; result and clock hashes were deterministic.

Three candidates passed every all-stage source gate:

- `HVRSSR-8__FILTERED_BY__HVTCCR-8`: 8/22/23/10 events,
- `HVRSSR-8__FILTERED_BY__HVLZC-8`: 33/40/26/17 events,
- `HVSVF-8__FILTERED_BY__HVLZC-8`: 32/39/40/10 events.

The other three are terminal source rejections, chiefly from final-period event or
month-concentration failure. They cannot be repaired or promoted. The three
eligible candidates advance to unchanged Gross9 novelty with economics sealed.

Deterministic support SHA-256:
`035ad63c5b75f3a143fa9337e09bea65ac2cc3c5db6e4eb1054b585bdc351762`.
