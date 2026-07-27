# PSIM-D8-RLLM2-S3 chunked-native source-feature seal success

Date: 2026-07-27

Status: **SOURCE-ONLY PASS — only 2020 train outcomes may now be opened**

## Result

The official S3 run executed from commit
`dfef2ee175f6c055da71dbc797d611db4355e921` with the exact preregistered
`google/gemma-4-E4B-it` revision
`ee0ef6023621cff504d758262d4e04895a5af4a2`.

It passed the pre-market repeatability/capacity gate and completed a fresh
third-load extraction:

- independent gate model loads: **2**;
- repeatability cases: **10 / 10 passed on both loads**;
- cross-load embedding byte hashes: **all identical**;
- cross-load relation-logit byte hashes and codes: **all identical**;
- row 341 capacity prompt: **29,727 / 29,728 tokens**;
- row 341 peak allocated VRAM: **10,449,585,664 / 10,451,617,280 bytes**;
- full extraction model load: **fresh load 3**;
- source rows embedded: **1,461 / 1,461**;
- relation forwards: **1,344**;
- chunk forwards: **12,772**;
- extraction peak allocated VRAM: **10,486,249,472 bytes**;
- market/funding rows parsed: **0 / 0**;
- rewards/economic metrics created: **0 / 0**; and
- 2020, test, and eval outcomes opened during S3: **false**.

This is a source-representation integrity result, not a profitability claim.
The terminal action authorizes opening **2020 train outcomes only** for the
next preregistered semantic-alpha training stage. Outcomes from 2021 onward
remain sealed.

## Exact authority evidence

- Executed runner SHA-256:
  `ab12b42f2f9f6cbad02ae85fd61511b333736bdae6028462af8052fe77ec72a9`
- Attempt SHA-256:
  `99046f2ef3d816c279136e61e7c60394b39b5adc5cc790735d38025585fd5bcb`
- Attempt hash:
  `9bc78f6774246a41c88c57ebbe855cb7c4f7577e9fff814b080403400497bf4f`
- Repeatability gate SHA-256:
  `075f19c855d49146cb38c1bb40bd6599389736f7fac9e1a998c0eac320e6db50`
- Repeatability gate result hash:
  `1fce8b64b68dff2b99d0d940a2fd2d02379d21c24e37ac79b0676561c190bceb`
- Terminal result SHA-256:
  `0278b303005ae50510004344eb8889bc464c4b61612e0e162e091e7decd8a976`
- Terminal result hash:
  `9bf7121d4feec0f7000626f8493e05effc3dfff8f9857b7ac5f05f6beadc3af9`
- Deterministic gzip execution log SHA-256:
  `8f45d4c08b0bf8747300ed4990893fba6d1dd277e474e6c42d16522acb1f36be`
- Decompressed raw execution log SHA-256:
  `8992814559f716f898894e8ed7486a6070a9a6779e701b74ddbd9cfdb829af35`

## Exact source artifacts

- Source rows:
  `2897845bf55506bf877c2015a670707287d3d98c3718361e24ad31504b98939c`
- Chunked-native embeddings:
  `509d7922561cdec0582165e5976ac9bdcc72dfe26e68c6ce46fe7ba9fab4f2a0`
- Relation logits:
  `384725cd8f2f451a8dedca76625333d66517937b744e09ea5d71cd938c500b62`
- Relation rows:
  `4ea96c6af2da8f574eed88fcbfa7fe6f20b5d6131e89eb41b734a0e498cfef1a`

All 1,461 embeddings are finite `float32` values with width 2,560. All 1,344
forwarded relation rows have finite six-code logits. The row-granular
checkpoint directory and publish journal were removed before the authorizing
result was published.

## Next authorized research step

Create a fresh preregistration that:

1. binds this exact S3 result and artifact hashes;
2. opens only 2020 train outcomes;
3. fixes the semantic target, training algorithm, hyperparameter roster, and
   checkpoint selection rule before scoring;
4. prohibits access to 2021–2023 outcomes during model selection; and
5. evaluates the frozen selected candidate sequentially on untouched 2021
   test, 2022 validation, and 2023 evaluation periods before any later market
   data is opened.
