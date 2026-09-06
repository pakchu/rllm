# HVFXTE-12 source rejection

HVFXTE-12 is terminally rejected unchanged at source support. It produced
`3/13/16/6` train/test/eval/final events. Direction balance passed in all four
splits, while train and final missed their frozen event minima and train's
maximum-month share was `0.66667`, above the `0.45` ceiling.

Two independent database extractions and builds reproduced every FX network
session, split and primary clock, diagnostic-control clock, source manifest, and
support report byte-for-byte. The primary clock SHA-256 is
`841916e40fde4c4e293e1e21b7d50f48510993d5e633910c02574a5388ebd6e5`
and the support report SHA-256 is
`aed6aadaa24d197c40741d954b07d72bffcc52fdd7cb61bc12847b8ca0682e0f`.

Gross9, execution prices, returns, PnL, economics, and RV20 were not opened. No
universe, session, entropy estimator, graph, breadth, rank, threshold, side,
clock, hold, subset, or diagnostic control repair is authorized.
