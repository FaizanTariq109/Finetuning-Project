# Transformer fine-tuning demos

University transformer applications by Faizan Tariq. This repository contains T5 summarization and ViT food-image inference source. Portfolio preparation is proceeding one app at a time; no public deployment is claimed yet.

## T5 News Summarizer

The recovered T5-small checkpoint loads with its original configuration and tokenizer and produces summaries in local smoke tests. The application now has relative-path handling, artifact checksums, cached CPU inference, bounded input/output and explicit truncation feedback.

- [Project README](t5-small-finetuned/ReadMe.md)
- [Local validation](t5-small-finetuned/VALIDATION.md)
- [Deployment guide](t5-small-finetuned/DEPLOYMENT.md)
- Entrypoint: `t5-small-finetuned/app.py`
- Dependencies: `t5-small-finetuned/requirements.txt`

The 242 MB recovered weight file stays outside normal Git history. The model and its card are published on Hugging Face; anonymous download, hashes and inference passed. GitHub approval and Streamlit Community Cloud deployment/testing remain pending. No benchmark score is presented as reproduced; the original training report is retained as historical evidence.

```bash
python -m venv .venv-t5
# Activate .venv-t5, then:
python -m pip install -r t5-small-finetuned/requirements.txt
python -m streamlit run t5-small-finetuned/app.py
```

## ViT classifier — deferred

`ViT-finetuned/app.py` and its original configuration/processor remain available. Its recovered weights have not been restored into this checkout. The saved configuration has 101 labels despite the historical Food41 UI name; model/dataset naming and inference require a separate review. Existing working-copy repairs are preserved. Do not present it as validated or deployed yet.

Use the app-specific requirements for the prepared T5 application.

## Provenance and configuration

T5/ViT are existing Hugging Face model architectures. The contribution is fine-tuning and application work, not inventing those base architectures. The owner confirmed individual authorship of the T5 fine-tuning and Streamlit project; its original training notebook is unavailable.

See `.env.example` for optional model settings. No `.env` is loaded automatically. Real secrets, environments, audit backups, caches and recovered weights are excluded from Git. The previously reported historical T5 API-key candidate was reviewed: it is a Transformers import-line false positive, not a credential.

No original university archive file or deployed portfolio website file was changed during preparation.
