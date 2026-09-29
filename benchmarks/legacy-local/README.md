# Historical local A100 evidence

`a100-history.json` preserves selected aggregate measurements from the separate local
`ourobors` implementation at commit `93c9043`. Each original report has its source SHA256.
No dataset, checkpoint or raw run output was imported.

These are **not current-repository benchmark results**. In particular, model sizes, encoder
implementations, tokenizer and torch versions differ. The resume pilot used ChEMBL-rendered
images (not real-document depictions), 256 training images and 32 validation images; its
100-step loss comparison is a numerical resume check, not an accuracy result. The 512-token
capacity probe ran only three optimizer steps and is not a full-run throughput estimate.

Use the commands in `docs/reliability.md` to measure this repository's configured models.
