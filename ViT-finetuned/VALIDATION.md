# Local validation — 2026-09-29

Scope: ViT only, within the existing Finetuning-Project umbrella repository. The immutable university archive was read, not modified. Existing working files were backed up under ignored `.audit/vit-preparation` before changes.

## Recovered evidence

- Archive weight: 343,528,508 bytes. SHA-256 `f705b57ae486d42cd55a7314b821c400c5f6c0f54b4b39b20dd6c787160335c1`.
- Archive, working and existing public Hub configuration/processor JSON are semantically identical. Differences in raw JSON file hashes are line endings.
- Existing public Hub weight metadata and a fresh anonymous download match the archive weight hash.
- Classifier tensor shape: 101 × 768. Parameter count: 85,876,325.
- All 101 labels and their indices exactly match ETH Zurich Food-101 metadata.
- Owner confirmed individual work and reported original `/kaggle/input/food41/images` input from Kaggle `kmader/food41`, titled Food Images (Food-101). The supplied path is provenance evidence, not a runtime dependency.
- No ViT notebook/training state/evaluation report was found in the recovered fine-tuning project. Exact upstream checkpoint, training settings, subset and split remain unverified.

## Checks completed

- Original checkpoint loads with the saved processor/config: no missing, unexpected or mismatched tensors.
- Four real Food-101 validation image smoke tests: beignets, bruschetta, carrot_cake and frozen_yogurt. Preprocessing shape 1×3×224×224; finite 101-class softmax; sums within 0.000001 of one; top-five labels map to saved IDs. Full examples appear in README. Not a benchmark; training overlap unknown.
- Separate Python 3.11.16 ViT environment installed: 55 packages; dependency compatibility check passed. Existing T5 environment was not modified.
- Linux x86_64 / Python 3.11 dependency resolution passed. Actual Linux execution remains the Community Cloud validation step.
- Anonymous cold Hub download plus checksum/load/inference: passed in approximately 26 seconds on this Windows machine. Cached reruns reuse the artifact. An initial local download stalled; a bounded HTTP retry succeeded. Timing is not a hosting guarantee.
- Browser: initial page, real JPG upload, uploaded-image preview, five predictions and saved labels verified. Top result for row 1000 was bruschetta, 99.94%. Very small probabilities display <0.01% instead of rounding to zero.
- Five regression tests passed: invalid/oversized images; RGB conversion from grayscale/RGBA; initial/invalid-input UI; sanitized load failure; explicit retry recovery with the same upload. The retry/error tests simulate failure; food-image inference uses the real checkpoint.
- Gitleaks ViT directory scan: zero findings, with redacted reporting. No API key required. Weight, environments and audit data are ignored by Git. No original local paths used for runtime.
- App working set after inference: approximately 638 MiB, peak approximately 650 MiB on Windows. This is not a Linux/Community Cloud memory guarantee.
- Root README modifications confined to ViT section. No T5 source/documentation changes. Existing untracked root requirements.txt is excluded from this proposed change.

## Run the tests

```bash
python -m unittest discover -s ViT-finetuned/tests -v
```

Unit tests require the project dependencies but do not download model weights. Real smoke-test images and raw diagnostic files remain local in the ignored audit directory. The user-facing screenshot in docs demonstrates the UI only.

## Remaining release steps

Owner approval before committing/pushing the reviewed ViT changes; then Streamlit Community Cloud deployment with Python 3.11 and `ViT-finetuned/app.py`. Verify public cold start, upload, inference and error behavior before marking deployment complete or adding a portfolio Live Demo URL. No weight upload is needed.
