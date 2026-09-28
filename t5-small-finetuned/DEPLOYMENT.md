# T5 deployment guide

Status: the owner published the model bundle; anonymous download, hashes and inference passed. The model card is published; source push approval and cloud deployment remain pending.

## 1. Published model bundle — complete

The owner published all seven verified inference files and the model card at https://huggingface.co/FaizanTariq109/t5-small-news-summarizer. Anonymous download, all artifact checksums and inference passed. The app's Hub-selection path also passed without adjacent local weights.

The immutable model-bundle revision is `1cc579fc6a968c116b2358a5a80d6c45d5baba64`. The later model-card revision is `07c39398919c184d8a62341294c7bb90df37a7ad`; the model-bundle pin remains valid.

Model files stay on Hugging Face, with source on GitHub. This avoids adding the 242 MB checkpoint to normal Git history, but still requires host memory and an initial network download. Real cloud cold-start behavior remains untested.

## 2. Approve and publish source changes

Use the existing repository `FaizanTariq109/Finetuning-Project`; preserve its history. The planned entrypoint is `t5-small-finetuned/app.py` on the existing `main` branch after review/approval. No new replacement repository is needed.

Before the first push, review the preparation summary: exact changed files, runtime fixes, documentation, exclusions, model-storage choice, security findings and proposed commit message. The owner must explicitly approve the first push. Existing unrelated ViT changes must not be staged with the T5 changes.

Model weights, `.venv*`, `.audit`, caches, local secrets and archive backups must remain untracked. Historical evaluation numbers are retained only in the clearly labeled original report; they are not advertised as validated metrics.

## 3. Create the Streamlit app

After the source push and public model verification:

1. The owner signs in to [Streamlit Community Cloud](https://share.streamlit.io/) using GitHub and selects the `FaizanTariq109` workspace. Complete any authorization prompts personally.
2. Choose **Create app**, then **Yup, I have an app** (labels can change).
3. Repository: `FaizanTariq109/Finetuning-Project`.
4. Branch: `main` (verify the approved commit is on that branch).
5. Main file: `t5-small-finetuned/app.py`.
6. Choose an available descriptive URL; do not record it as live until tested.
7. In **Advanced settings**, explicitly select **Python 3.11**. Do not accept a different default without revalidation.
8. Enter these root-level Streamlit settings, substituting the verified repository and SHA:

```toml
T5_MODEL_REPO = "FaizanTariq109/t5-small-news-summarizer"
T5_MODEL_REVISION = "1cc579fc6a968c116b2358a5a80d6c45d5baba64"
```

Leave `T5_MODEL_DIR` unset on the cloud. Public download needs no API key or HF access token. Never copy the Windows archive/working-directory path into cloud settings. For ordinary environment variables outside Community Cloud, the same two names apply. See `.env.example` for local configuration; it is not read automatically.

9. Check that Streamlit installs **the requirements beside the entrypoint**, `t5-small-finetuned/requirements.txt`. These are the prepared CPU dependencies for this app.
10. Click **Deploy** and watch the build log. Stop for the owner whenever a login, authorization, or account/browser action is required.

## 4. Test the public URL

- Open the app anonymously and verify the initial UI renders.
- Use the example and generate a non-empty, relevant summary.
- Test an empty article, a longer-than-512-token article, and a second independent article.
- Change generation settings, rerun and download the summary.
- Open a second session to confirm requests are handled without crashes.
- Reboot through Manage app; test another generation. A fresh cache may redownload model artifacts.
- Inspect cloud logs and resource behavior. Record the final public URL and checks in `AI-DEPLOYMENT-TRACKER.md` only after they pass.

The local machine's timing/memory measurements do not guarantee cloud performance. Community Cloud documentation gives approximate, changeable resource limits, not a dedicated hardware guarantee. If repeated resource failures prevent reliable use, first examine the logs and dependency footprint. A Streamlit Docker Space on Hugging Face is a fallback to assess, not an automatic migration.

## Troubleshooting

| Symptom | Check |
|---|---|
| Dependency build failure | Confirm Python 3.11, app-specific requirements and CPU wheel availability in the log. Do not add CUDA packages. |
| Model configuration error | Repository must be public; revision must be the full commit SHA, with all seven expected files. |
| Checksum failure | Use the exact recovered files. Do not bypass the manifest or substitute base weights. |
| Download/network error | Retry after checking model-repository availability; cache is reused when the snapshot is complete. |
| Slow first generation | Distinguish import/download/model-load time from subsequent inference time. |
| Memory/resource error | Inspect Manage app logs and repeated-generation behavior before changing hosting. |
| Empty/poor summary | Try a complete English news article. Short inputs and out-of-domain text are not validated. |
| Import path error | Launch the exact entrypoint shown above. Model paths resolve against its directory. |

## Official references checked during preparation

- [Deploy an app and choose Python/settings](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [App dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Manage logs, restarts and resource limits](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app)
- [Hugging Face public storage policy](https://huggingface.co/docs/hub/storage-limits)

A public URL opening successfully is only the first check; generation and restart behavior must also be tested before marking the project portfolio-ready.
