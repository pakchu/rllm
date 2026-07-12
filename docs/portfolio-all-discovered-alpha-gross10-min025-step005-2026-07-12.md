# All discovered alpha portfolio — gross 10 / min 0.25 / step 0.05 (2026-07-12)

Weights ranked on test2024 only; eval2025/ytd2026 report-only. Gross<=10, nonzero weight>=0.25, step=0.05, cost=6bp/side, strict MDD includes adverse excursion.

This is a broad research portfolio: weight selection is test2024-only, but the alpha universe includes candidates discovered while examining later windows. Treat 2025/2026 as diagnostics, not pristine final eval.

Evaluated: 11433; sleeves=381; extra_candidate_sleeves=347.
Metric: `absolute return / full-window CAGR / strict MDD / CAGR-MDD / trades`.

## Top selected by 2024 test only
|#|gross|weights|train|test2024|eval2025|2026YTD|
|---:|---:|---|---:|---:|---:|---:|
|1|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_kimchi_gate_0': 3.0}`|213.16/40.86/78.90/0.52/2013|368.78/367.30/12.93/28.40/348|378.89/379.40/10.93/34.72/273|31.52/92.32/21.29/4.34/183|
|2|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_kimchi_gate_1': 3.0}`|134.59/29.16/79.63/0.37/2031|353.17/351.76/12.93/27.20/352|376.64/377.15/12.18/30.97/276|30.95/90.34/21.29/4.24/184|
|3|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_state_funding_relief_vs_fx_stress': 3.0}`|145.02/30.86/79.29/0.39/2045|331.61/330.32/13.01/25.39/358|378.85/379.37/11.96/31.73/276|27.25/77.77/22.53/3.45/185|
|4|8.00|`{'new_long_range_funding_premium': 1.6, 'cand_path_gate_0': 1.6, 'cand_jump_volume_gate_2': 2.0, 'cand_calendar_1_caloi_c9e5160375_long_h48_s6': 0.9, 'cand_calendar_40_caloi_8d33aa9e0c_short_h48_s24': 1.6, 'cand_rex_veto_54': 0.3}`|70.51/17.37/54.97/0.32/894|157.84/157.34/6.25/25.19/194|79.59/79.67/9.28/8.58/148|14.32/37.64/13.27/2.84/121|
|5|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_kimchi_gate_2': 3.0}`|115.49/25.91/79.83/0.32/2017|308.85/307.67/12.65/24.33/346|371.30/371.80/13.15/28.26/274|34.19/101.79/21.29/4.78/182|
|6|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_kimchi_gate_4': 3.0}`|139.84/30.02/76.34/0.39/2013|296.95/295.83/12.54/23.58/347|400.95/401.50/10.93/36.74/272|34.97/104.61/21.29/4.91/181|
|7|5.00|`{'oi_high_sel': 1.15, 'new_long_range_funding_premium': 3.05, 'cand_calendar_235_caloi_fb3665062a_long_h144_s24': 0.8}`|12.52/3.60/88.93/0.04/1209|403.60/401.94/17.09/23.52/238|142.22/142.37/12.89/11.04/162|29.62/85.77/16.74/5.12/78|
|8|6.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_kimchi_gate_0': 2.0}`|123.47/27.29/78.66/0.35/2013|288.08/287.01/12.29/23.35/348|321.29/321.70/9.60/33.50/273|33.38/98.91/17.81/5.55/183|
|9|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_kimchi_gate_5': 3.0}`|116.18/26.03/75.77/0.34/2009|286.26/285.19/12.54/22.73/350|412.54/413.12/11.96/34.55/273|35.55/106.69/21.29/5.01/181|
|10|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_kimchi_gate_7': 3.0}`|214.27/41.01/78.62/0.52/2019|290.14/289.05/12.93/22.35/347|379.10/379.61/10.93/34.74/271|37.08/112.30/21.29/5.28/182|
|11|6.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_kimchi_gate_1': 2.0}`|84.90/20.26/78.89/0.26/2031|279.68/278.64/12.63/22.05/352|320.14/320.55/10.60/30.24/276|33.01/97.59/17.81/5.48/184|
|12|10.00|`{'new_long_range_funding_premium': 3.25, 'cand_jump_volume_gate_9': 5.95, 'cand_calendar_243_caloi_9683a3c8ab_long_h144_s24': 0.8}`|10.09/2.93/83.18/0.04/476|389.36/387.77/17.65/21.97/89|132.14/132.28/15.48/8.55/75|17.45/46.80/25.01/1.87/68|
|13|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_path_gate_0': 3.0}`|10.42/3.02/84.47/0.04/1820|269.83/268.84/12.31/21.84/320|343.17/343.62/11.96/28.74/236|26.23/74.39/17.26/4.31/135|
|14|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_path_gate_1': 3.0}`|10.42/3.02/84.47/0.04/1820|269.83/268.84/12.31/21.84/320|343.17/343.62/11.96/28.74/236|26.23/74.39/17.26/4.31/135|
|15|5.75|`{'pb30_base': 0.25, 'nonpb30_taker': 1.0, 'oi_raw': 0.5, 'rex_rule': 2.0, 'short_premium_panic': 2.0}`|-33.73/-11.62/84.58/-0.14/1934|398.22/396.58/18.49/21.45/340|199.11/199.34/9.24/21.57/235|52.63/174.42/12.92/13.50/114|
|16|6.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_state_funding_relief_vs_fx_stress': 2.0}`|91.21/21.48/78.86/0.27/2045|267.70/266.72/12.51/21.31/358|321.57/321.99/10.45/30.81/276|30.48/88.74/17.81/4.98/185|
|17|6.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_kimchi_gate_2': 2.0}`|74.19/18.12/79.26/0.23/2017|254.13/253.22/12.29/20.60/346|316.97/317.38/11.26/28.18/274|35.11/105.12/17.81/5.90/182|
|18|8.00|`{'pb30_base': 0.55, 'new_long_funding_compression_premium': 2.65, 'cand_macro_long_0_usdkrw_relief_weekly': 2.45, 'cand_macro_long_1_usdkrw_relief_weekly': 0.75, 'cand_calendar_203_caloi_a8eac0e7b5_long_h72_s24': 0.35, 'cand_calendar_219_caloi_df18dba79b_short_h144_s12': 1.25}`|542.59/74.78/56.78/1.32/915|309.86/308.67/15.13/20.40/162|74.67/74.73/12.71/5.88/135|59.37/204.24/11.17/18.28/80|
|19|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_macro_short_0_usdkrw_riskoff_weakness': 3.0}`|57.20/14.54/73.89/0.20/1844|232.33/231.52/11.48/20.17/323|265.63/265.95/11.54/23.05/233|28.33/81.39/15.78/5.16/115|
|20|6.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_kimchi_gate_4': 2.0}`|86.93/20.65/76.96/0.27/2013|247.26/246.37/12.29/20.04/347|334.05/334.49/9.60/34.83/272|35.63/107.02/17.81/6.01/181|
|21|6.10|`{'nonpb30_taker': 0.65, 'oi_raw': 0.55, 'rex_rule': 2.0, 'oi_upbit_ratio288_low': 2.25, 'bear_rex_short': 0.3, 'oi_alt_ratio72_dyn_exit': 0.35}`|-50.01/-18.79/85.30/-0.22/2950|390.71/389.11/19.71/19.74/489|355.20/355.68/13.19/26.96/308|30.96/90.40/16.58/5.45/143|
|22|6.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_kimchi_gate_7': 2.0}`|124.11/27.40/78.49/0.35/2019|243.29/242.42/12.29/19.72/347|321.43/321.85/9.60/33.51/271|37.08/112.31/17.81/6.30/182|
|23|6.00|`{'short_premium_panic': 0.8, 'oi_high_sel': 0.75, 'cand_macro_long_2_usdkrw_relief_weekly': 3.35, 'cand_calendar_107_caloi_d602b24d54_long_h72_s24': 1.1}`|-62.32/-25.39/70.45/-0.36/1329|104.22/103.92/5.27/19.72/266|19.64/19.66/9.05/2.17/182|9.15/23.23/6.74/3.45/71|
|24|9.00|`{'new_long_funding_compression_premium': 2.9, 'cand_path_gate_0': 1.3, 'cand_jump_volume_gate_9': 3.35, 'cand_calendar_209_caloi_1465aae38c_long_h96_s24': 0.35, 'cand_rex_veto_11': 1.1}`|675.01/84.89/66.92/1.27/876|351.67/350.28/17.91/19.56/179|115.29/115.41/15.16/7.61/172|40.46/125.03/18.09/6.91/118|
|25|6.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_kimchi_gate_5': 2.0}`|74.42/18.17/76.22/0.24/2009|241.15/240.29/12.29/19.55/350|340.78/341.23/10.45/32.66/273|36.02/108.42/17.81/6.09/181|

## Robust diagnostic only (eval-influenced)
|#|gross|weights|test2024|eval2025|2026YTD|
|---:|---:|---|---:|---:|---:|
|1|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_kimchi_gate_6': 3.0}`|260.90/259.95/17.70/14.69/334|326.69/327.11/11.96/27.36/237|64.59/228.55/16.22/14.09/138|
|2|5.75|`{'pb30_base': 0.25, 'nonpb30_taker': 1.0, 'oi_raw': 0.5, 'rex_rule': 2.0, 'short_premium_panic': 2.0}`|398.22/396.58/18.49/21.45/340|199.11/199.34/9.24/21.57/235|52.63/174.42/12.92/13.50/114|
|3|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_calendar_206_caloi_d602b24d54_long_h144_s24': 3.0}`|232.62/231.80/18.06/12.83/326|187.51/187.71/14.10/13.31/251|62.38/218.13/16.81/12.97/117|
|4|6.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_kimchi_gate_6': 2.0}`|225.48/224.69/17.59/12.78/334|288.59/288.95/10.45/27.65/237|53.98/180.26/15.92/11.33/138|
|5|6.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_calendar_206_caloi_d602b24d54_long_h144_s24': 2.0}`|207.35/206.64/17.20/12.02/326|198.71/198.93/10.64/18.69/251|52.16/172.40/16.46/10.48/117|
|6|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_calendar_138_caloi_4ccc23caa6_long_h96_s6': 3.0}`|197.72/197.05/19.24/10.24/332|193.41/193.63/11.70/16.56/243|47.27/151.95/14.47/10.50/115|
|7|4.15|`{'nonpb30_taker': 0.75, 'oi_raw': 0.4, 'rex_rule': 1.5, 'short_premium_panic': 1.5}`|224.93/224.14/14.00/16.01/320|117.49/117.61/7.37/15.96/215|35.51/106.55/10.54/10.11/97|
|8|6.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_calendar_73_caloi_fcfd53ee04_long_h144_s12': 2.0}`|161.88/161.36/13.75/11.74/330|149.65/149.81/12.25/12.23/241|45.16/143.45/15.04/9.54/120|
|9|5.45|`{'nonpb30_taker': 1.7, 'rex_rule': 2.1, 'oi_wave_lowpos144': 0.4, 'oi_high_sel': 0.4, 'rex_dyn_short_exit': 0.85}`|266.20/265.23/19.70/13.46/387|273.18/273.51/10.94/24.99/289|48.70/157.85/16.86/9.36/128|
|10|2.75|`{'nonpb30_taker': 0.5, 'oi_raw': 0.25, 'rex_rule': 1.0, 'short_premium_panic': 1.0}`|121.07/120.71/9.36/12.90/320|68.09/68.15/5.03/13.54/215|23.36/65.08/7.03/9.26/97|
|11|6.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_calendar_5_caloi_d227bd538a_long_h144_s6': 2.0}`|166.59/166.05/17.30/9.60/320|232.31/232.58/10.74/21.65/239|45.71/145.64/15.78/9.23/113|
|12|6.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_calendar_138_caloi_4ccc23caa6_long_h96_s6': 2.0}`|185.40/184.79/17.04/10.85/332|202.54/202.77/9.60/21.11/243|42.45/132.72/14.47/9.17/115|
|13|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_calendar_54_caloi_6a645c6ed6_long_h72_s6': 3.0}`|178.20/177.62/14.78/12.01/314|227.58/227.84/10.53/21.63/223|45.17/143.48/15.78/9.09/113|
|14|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_calendar_5_caloi_d227bd538a_long_h144_s6': 3.0}`|168.45/167.91/18.55/9.05/320|237.38/237.66/13.78/17.25/239|52.28/172.91/15.78/10.96/113|
|15|4.60|`{'rex_rule': 0.55, 'new_long_funding_compression_premium': 1.55, 'cand_jump_volume_gate_14': 0.55, 'cand_calendar_28_caloi_c9e5160375_long_h96_s6': 0.65, 'cand_calendar_41_caloi_4d2149d868_short_h144_s12': 0.45, 'cand_calendar_68_caloi_6cb4c38b7f_short_h96_s6': 0.4, 'cand_calendar_160_caloi_a59814d430_long_h72_s24': 0.45}`|136.11/135.70/10.40/13.05/259|64.21/64.26/7.19/8.93/238|31.68/92.91/5.44/17.07/135|
|16|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_calendar_72_caloi_c44c3d2250_long_h96_s6': 3.0}`|176.67/176.09/15.81/11.14/315|225.45/225.71/12.39/18.21/235|49.15/159.69/17.96/8.89/114|
|17|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_calendar_22_caloi_04ed57f8a2_long_h144_s6': 3.0}`|196.79/196.13/18.55/10.57/329|249.17/249.47/15.26/16.35/254|48.46/156.86/17.82/8.80/118|
|18|5.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_kimchi_gate_6': 1.0}`|191.99/191.35/17.48/10.95/334|252.94/253.25/9.60/26.37/237|43.45/136.64/15.83/8.63/138|
|19|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_calendar_172_caloi_d602b24d54_long_h72_s6': 3.0}`|165.58/165.05/17.16/9.62/338|156.75/156.91/12.33/12.72/256|47.64/153.48/17.81/8.62/120|
|20|5.25|`{'nonpb30_taker': 1.2, 'oi_raw': 0.4, 'rex_rule': 2.05, 'oi_wave_lowpos144': 0.4, 'oi_upbit_ratio288_low': 0.4, 'rex_dyn_short_exit': 0.8}`|257.98/257.04/19.82/12.97/471|262.23/262.55/10.55/24.89/331|40.49/125.15/14.55/8.60/143|
|21|6.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_calendar_207_caloi_7c666514ac_long_h144_s6': 2.0}`|147.97/147.51/17.18/8.58/368|193.03/193.25/14.15/13.66/278|39.59/121.72/12.52/9.72/127|
|22|6.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_calendar_211_caloi_8cdaaed710_long_h96_s6': 2.0}`|167.64/167.10/19.56/8.54/359|174.97/175.16/12.77/13.72/270|44.93/142.51/14.47/9.85/127|
|23|3.00|`{'new_long_funding_compression_premium': 0.6, 'new_long_range_funding_premium': 1.25, 'new_short_premium_kimchi_union': 1.15}`|145.80/145.35/8.27/17.57/124|67.77/67.83/7.98/8.49/157|37.76/114.83/6.39/17.96/93|
|24|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_calendar_205_caloi_5126a6d0fb_long_h96_s12': 3.0}`|185.82/185.20/15.27/12.13/317|231.96/232.23/9.60/24.18/226|39.55/121.55/14.47/8.40/113|
|25|7.50|`{'nonpb30_taker': 1.0, 'rex_rule': 1.0, 'oi_high_sel': 1.0, 'bear_rex_short': 1.5, 'cand_calendar_65_caloi_1ee52c6a8a_long_h72_s24': 3.0}`|162.86/162.34/15.58/10.42/316|227.83/228.10/9.60/23.75/232|39.52/121.44/14.47/8.39/114|

## Train-sane diagnostic
Ranks rows that also prefer train strict MDD<=20. Cell format is unchanged.
|#|gross|weights|train|test2024|eval2025|2026YTD|
|---:|---:|---|---:|---:|---:|---:|
|1|3.00|`{'new_long_minimal_funding_premium': 1.0, 'cand_calendar_5_caloi_d227bd538a_long_h144_s6': 0.25, 'cand_calendar_72_caloi_c44c3d2250_long_h96_s6': 0.7, 'cand_calendar_146_caloi_48b8800ece_short_h48_s12': 0.75, 'cand_rex_veto_21': 0.3}`|121.06/26.88/17.31/1.55/627|31.56/31.48/6.45/4.88/137|22.39/22.40/6.04/3.71/139|16.38/43.64/5.11/8.54/78|
|2|3.50|`{'new_long_minimal_funding_premium': 0.6, 'cand_calendar_44_caloi_c18028a970_long_h144_s6': 0.8, 'cand_calendar_131_caloi_4a81b22793_long_h96_s12': 0.4, 'cand_calendar_166_caloi_dba4ab9784_short_h144_s12': 0.7, 'cand_calendar_216_caloi_403820a3ff_long_h72_s12': 0.4, 'cand_rex_veto_27': 0.6}`|94.94/22.18/17.21/1.29/488|16.14/16.10/4.99/3.22/102|14.85/14.86/4.81/3.09/118|7.18/18.00/4.11/4.38/66|
|3|2.80|`{'rex_wave_vol144_high': 0.25, 'new_long_range_funding_premium': 0.55, 'cand_jump_volume_gate_12': 0.55, 'cand_calendar_98_caloi_2c9d972e93_long_h96_s12': 0.55, 'cand_rex_veto_17': 0.9}`|82.69/19.82/18.18/1.09/1019|25.45/25.39/6.89/3.69/213|17.13/17.14/4.86/3.52/167|8.56/21.67/4.52/4.80/110|
|4|2.60|`{'cand_kimchi_gate_6': 1.45, 'cand_jump_volume_gate_18': 0.75, 'cand_rex_veto_18': 0.4}`|49.15/12.75/14.30/0.89/734|22.54/22.49/5.94/3.78/149|23.71/23.73/6.77/3.51/102|12.54/32.57/7.78/4.19/97|
|5|2.65|`{'oi_vol_alt_ratio288': 0.5, 'new_long_range_funding_premium': 0.55, 'cand_jump_volume_gate_2': 0.3, 'cand_calendar_168_caloi_fb123122e2_short_h96_s12': 0.6, 'cand_calendar_241_caloi_4a81b22793_long_h96_s6': 0.4, 'cand_rex_veto_5': 0.3}`|43.12/11.36/15.16/0.75/917|33.39/33.31/3.89/8.57/251|17.03/17.04/4.87/3.50/188|7.69/19.35/3.45/5.61/103|
|6|2.05|`{'new_long_range_funding_premium': 0.6, 'cand_calendar_26_caloi_0d761bc016_long_h144_s6': 0.6, 'cand_calendar_234_caloi_aa2c2f4d40_short_h144_s12': 0.5, 'cand_rex_veto_24': 0.35}`|42.55/11.23/16.36/0.69/1045|25.01/24.95/3.91/6.38/247|14.24/14.25/4.43/3.22/249|13.11/34.18/2.22/15.42/134|
|7|4.00|`{'cand_path_gate_2': 1.0, 'cand_macro_short_0_usdkrw_riskoff_weakness': 1.2, 'cand_calendar_118_caloi_7040ef2152_long_h144_s12': 1.8}`|116.98/26.17/9.10/2.87/296|27.86/27.80/4.49/6.19/63|8.47/8.48/7.80/1.09/86|-3.58/-8.32/10.37/-0.80/73|
|8|7.85|`{'pb30_base': 0.5, 'cand_path_gate_2': 2.6, 'cand_rex_veto_38': 2.9, 'cand_rex_veto_50': 1.85}`|1189.62/115.41/42.78/2.70/920|28.42/28.35/29.89/0.95/172|82.90/82.97/32.08/2.59/152|-7.95/-17.94/21.62/-0.83/120|
|9|7.80|`{'short_premium_panic': 0.3, 'new_long_minimal_funding_premium': 2.2, 'cand_calendar_1_caloi_c9e5160375_long_h48_s6': 0.3, 'cand_calendar_108_caloi_59af070cbf_long_h96_s6': 0.3, 'cand_rex_veto_16': 2.2, 'cand_rex_veto_33': 1.8, 'cand_rex_veto_38': 0.7}`|1712.55/138.58/52.80/2.62/1276|35.49/35.41/30.80/1.15/276|71.82/71.88/14.40/4.99/203|33.30/98.63/18.70/5.27/120|
|10|5.80|`{'cand_path_gate_2': 1.45, 'cand_jump_volume_gate_18': 0.75, 'cand_rex_veto_2': 2.05, 'cand_rex_veto_50': 1.55}`|511.71/72.21/27.81/2.60/1034|-9.59/-9.58/33.50/-0.29/210|41.57/41.60/26.34/1.58/174|-6.08/-13.91/16.06/-0.87/146|
|11|8.00|`{'new_long_minimal_funding_premium': 1.7, 'cand_kimchi_gate_6': 1.15, 'cand_calendar_152_caloi_8cdaaed710_long_h96_s12': 0.8, 'cand_rex_veto_5': 0.95, 'cand_rex_veto_28': 1.9, 'cand_rex_veto_43': 1.5}`|1439.07/127.16/49.33/2.58/1376|28.70/28.63/33.63/0.85/308|85.15/85.23/17.44/4.89/240|43.04/135.04/17.90/7.55/153|
|12|10.00|`{'oi_vol_alt_ratio288': 0.35, 'new_long_funding_compression_premium': 4.75, 'cand_rex_veto_25': 4.9}`|3618.10/196.00/76.63/2.56/523|381.04/379.50/39.14/9.70/163|155.75/155.92/23.99/6.50/138|96.54/401.82/21.32/18.85/71|
|13|5.90|`{'rex_wave_vol144_high': 1.25, 'cand_kimchi_gate_0': 1.65, 'cand_calendar_150_caloi_d602b24d54_long_h72_s12': 1.05, 'cand_rex_veto_2': 0.8, 'cand_rex_veto_18': 1.15}`|767.52/91.25/36.11/2.53/1106|58.16/58.01/18.35/3.16/229|53.71/53.75/11.97/4.49/196|9.63/24.55/16.59/1.48/146|
|14|2.90|`{'cand_path_gate_2': 0.65, 'cand_jump_volume_gate_15': 0.35, 'cand_calendar_30_caloi_ede6db231a_long_h144_s24': 0.4, 'cand_rex_veto_10': 1.5}`|176.61/35.71/14.32/2.49/833|-8.65/-8.64/18.99/-0.45/170|19.09/19.11/9.51/2.01/152|-1.15/-2.72/7.23/-0.38/135|
|15|9.05|`{'new_long_minimal_funding_premium': 3.9, 'cand_calendar_16_caloi_0c016015db_long_h144_s12': 1.15, 'cand_rex_veto_56': 4.0}`|1912.39/146.19/60.72/2.41/548|66.40/66.23/41.51/1.60/108|121.03/121.15/19.97/6.07/96|53.41/177.76/19.40/9.16/64|
|16|7.85|`{'cand_macro_short_0_usdkrw_riskoff_weakness': 3.2, 'cand_calendar_105_caloi_ede6db231a_long_h96_s12': 0.45, 'cand_rex_veto_7': 3.2, 'cand_rex_veto_27': 1.0}`|837.64/95.76/39.80/2.41/576|-4.51/-4.50/32.29/-0.14/121|26.89/26.91/20.05/1.34/105|-4.15/-9.61/17.47/-0.55/57|
|17|10.00|`{'new_long_minimal_funding_premium': 2.85, 'cand_path_gate_1': 3.4, 'cand_rex_veto_15': 3.75}`|2072.14/151.90/64.54/2.35/541|158.67/158.17/31.20/5.07/113|230.68/230.96/20.86/11.07/96|29.41/85.05/18.14/4.69/84|
|18|3.85|`{'oi_upbit_ratio288_low': 0.65, 'new_long_minimal_funding_premium': 1.75, 'cand_rex_veto_7': 1.45}`|523.60/73.21/31.90/2.30/818|66.94/66.76/13.88/4.81/172|61.20/61.25/10.01/6.12/109|24.89/70.00/7.27/9.63/65|
|19|6.80|`{'pb30_base': 1.4, 'new_long_range_funding_premium': 0.3, 'cand_calendar_126_caloi_0c016015db_long_h72_s24': 0.8, 'cand_calendar_127_caloi_4a81b22793_long_h72_s12': 0.3, 'cand_calendar_167_caloi_f22ff24d6b_long_h96_s24': 0.35, 'cand_rex_veto_11': 1.4, 'cand_rex_veto_16': 0.65, 'cand_rex_veto_41': 1.6}`|957.53/102.96/45.04/2.29/1458|31.79/31.71/20.31/1.56/278|115.21/115.32/12.48/9.24/232|23.88/66.75/22.93/2.91/131|
|20|8.05|`{'rex_wave_vol144_high': 0.5, 'oi_upbit_ratio288_low': 0.25, 'cand_path_gate_2': 1.95, 'cand_calendar_54_caloi_6a645c6ed6_long_h72_s6': 0.45, 'cand_calendar_78_caloi_8d33aa9e0c_short_h24_s24': 2.5, 'cand_calendar_81_caloi_f22ff24d6b_long_h96_s12': 0.45, 'cand_rex_veto_39': 1.95}`|408.62/62.93/27.85/2.26/1168|50.61/50.49/17.90/2.82/272|38.11/38.15/14.10/2.71/207|-5.14/-11.83/15.09/-0.78/131|
|21|9.95|`{'cand_calendar_141_caloi_a2e67b5a09_short_h24_s12': 0.4, 'cand_calendar_241_caloi_4a81b22793_long_h96_s6': 3.55, 'cand_rex_veto_2': 6.0}`|1631.80/135.34/60.90/2.22/385|-47.10/-47.03/67.96/-0.69/98|27.83/27.86/32.16/0.87/79|13.23/34.54/26.41/1.31/33|
|22|5.75|`{'new_long_minimal_funding_premium': 2.05, 'cand_calendar_22_caloi_04ed57f8a2_long_h144_s6': 0.4, 'cand_calendar_145_caloi_ede6db231a_long_h72_s24': 0.35, 'cand_rex_veto_8': 1.8, 'cand_rex_veto_32': 1.15}`|830.07/95.29/43.38/2.20/731|72.41/72.22/13.71/5.27/160|52.66/52.70/10.77/4.89/148|18.66/50.43/13.58/3.71/90|
|23|7.95|`{'new_long_funding_compression_premium': 2.0, 'cand_rex_veto_12': 1.8, 'cand_rex_veto_36': 2.15, 'cand_rex_veto_43': 2.0}`|1680.54/137.31/63.28/2.17/1095|40.68/40.58/51.37/0.79/230|111.73/111.84/20.56/5.44/187|44.38/140.30/18.70/7.50/106|
|24|7.80|`{'pb30_base': 0.65, 'cand_calendar_31_caloi_c9e5160375_long_h72_s6': 2.2, 'cand_calendar_182_caloi_04ed57f8a2_long_h48_s24': 2.05, 'cand_rex_veto_4': 0.85, 'cand_rex_veto_16': 2.05}`|470.02/68.60/31.67/2.17/879|7.23/7.21/26.40/0.27/168|48.17/48.21/11.07/4.36/142|16.94/45.29/11.40/3.97/69|
|25|4.00|`{'new_long_minimal_funding_premium': 1.3, 'cand_path_gate_2': 0.25, 'cand_calendar_28_caloi_c9e5160375_long_h96_s6': 0.6, 'cand_rex_veto_13': 0.7, 'cand_rex_veto_43': 1.15}`|437.47/65.65/30.82/2.13/980|37.92/37.83/13.26/2.85/217|44.23/44.26/10.87/4.07/170|17.22/46.14/9.13/5.05/141|

## Candidate coverage
```json
{
  "extra_sleeves": 347,
  "event_counts_summary": {
    "train": 328,
    "test2024": 328,
    "eval2025": 325,
    "ytd2026": 325
  }
}
```
