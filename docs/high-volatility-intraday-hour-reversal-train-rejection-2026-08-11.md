# HVIHR-1 train rejection — 2026-08-11

The published `r3 -> r19` reversal pair did not transfer to the frozen 2023H2 high-variation Binance-perpetual implementation.

- Trades: `29` (`15` long, `14` short).
- Base absolute return: `-2.245%`; full-calendar CAGR: `-4.406%`; strict MDD: `3.074%`.
- Mean gross underlying move: `-3.57 bp`, below the required `20 bp`.
- Cluster sign-flip p-value: `0.9355`, above the required `0.10`.
- Stress absolute return: `-3.374%`.
- First half was `+0.107%`; second half was `-2.349%`, so the both-halves gate failed.
- Test, eval, final, and RV20-q90 audit outcomes remained unopened.
- Terminal action: reject unchanged. No hour-pair, threshold, side, entry, exit, or hold repair is authorized.
