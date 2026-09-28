# Transformer fine-tuning demos

University transformer applications by Faizan Tariq. This umbrella repository covers T5 news summarization, ViT food-image classification and GPT-2 recipe generation. T5 is publicly deployed; ViT is locally validated and prepared for deployment; GPT-2 preparation remains pending.

## T5 News Summarizer

The recovered T5-small checkpoint loads with its original configuration and tokenizer and produces summaries in local smoke tests. The application now has relative-path handling, artifact checksums, cached CPU inference, bounded input/output and explicit truncation feedback.

- [Live Demo](https://faizan-t5-summarizer.streamlit.app/)
- [Project README](t5-small-finetuned/ReadMe.md)
- [Local validation](t5-small-finetuned/VALIDATION.md)
- [Deployment guide](t5-small-finetuned/DEPLOYMENT.md)
- Entrypoint: `t5-small-finetuned/app.py`
- Dependencies: `t5-small-finetuned/requirements.txt`

The 242 MB recovered weight file stays outside normal Git history. The model and its card are published on Hugging Face; anonymous download, hashes and inference passed. The T5 application is published on GitHub and successfully deployed and tested on Streamlit Community Cloud. No benchmark score is presented as reproduced; the original training report is retained as historical evidence.

```bash
python -m venv .venv-t5
# Activate .venv-t5, then:
python -m pip install -r t5-small-finetuned/requirements.txt
python -m streamlit run t5-small-finetuned/app.py
```

## ViT food classifier - locally validated, deployment pending

The recovered checkpoint has **101 Food-101 labels**; `food41` is the original Kaggle dataset slug, not the class count. Local model loading and four food-image inference smoke tests passed. These are not benchmark results.

- [ViT documentation](ViT-finetuned/README.md)
- Entrypoint: `ViT-finetuned/app.py`
- Dependencies: `ViT-finetuned/requirements.txt`
- Weights: existing public `FaizanTariq109/ViTFinetuned` Hugging Face checkpoint, pinned and checksum-verified; no large weight in Git.
- Prepared and locally validated; not yet publicly deployed. Streamlit Community Cloud deployment and public verification remain pending.

Use each application's own requirements file.

## GPT-2 recipe generator - preparation pending

GPT-2 will be reviewed and prepared in a later task. No validation or deployment status is claimed here.

## Provenance and configuration

T5/ViT are existing Hugging Face model architectures. The contribution is fine-tuning and application work, not inventing those base architectures. The owner confirmed individual authorship of the T5 fine-tuning and Streamlit project; its original training notebook is unavailable.

See `.env.example` for optional model settings. No `.env` is loaded automatically. Real secrets, environments, audit backups, caches and recovered weights are excluded from Git. The previously reported historical T5 API-key candidate was reviewed: it is a Transformers import-line false positive, not a credential.

No original university archive file or deployed portfolio website file was changed during preparation.
