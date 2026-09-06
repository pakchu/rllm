# HVPASR-12 source-support pass — 2026-08-09

HVPASR-12 passed every frozen source gate without opening BTC execution prices,
post-entry returns, funding PnL, economic metrics, or Gross9 rows.

| Stage | Events | Long | Short | Minority share | Max month share |
|---|---:|---:|---:|---:|---:|
| train | 34 | 22 | 12 | 0.3529 | 0.3235 |
| test | 99 | 53 | 46 | 0.4646 | 0.2424 |
| eval | 78 | 33 | 45 | 0.4231 | 0.2436 |
| final | 56 | 21 | 35 | 0.3750 | 0.3393 |

All 8/12/12/8 event floors, the 0.20 minority-side floor, and the 0.45
single-month ceiling pass. The unchanged singleton may advance only to Gross9
structural novelty.

- support result SHA256: `b29d5562edc70149f2a7bd5992acfc8c64d4a8761724bd14430fc3a54296181c`
- support manifest hash: `7a96481b79a4add7a1ce278b77e84ff4cb6eed4f1b1a9296442a097e664dc1a8`
- primary clock SHA256: `83b2f8d9bc6423b4b73ab799f5e1112d0f983ef824c8d242c06d69d090d27e27`
- source feature SHA256: `52c590788aeecdae03b9821906b6b9061ba9ef6c33ee3115452fe7ea1bc8cb78`
