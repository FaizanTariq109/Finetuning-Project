# T5 local validation

Validated on 2026-09-28 on Windows with Python 3.11 in a separate `.venv-t5` environment. Public deployment is pending.

## Recovered model

The 242,041,896-byte checkpoint matches the archive byte-for-byte by SHA-256:
`881ecd3d2e20cfb1ddf96ac598ad90a610007dfa4ff04cfc024e4afd118bc686`.
All seven model/configuration/tokenizer files are checked against `model-manifest.json`. The model loaded offline with no missing, unexpected or mismatched state keys. Parameter count: 60,506,624.

## Checks completed

- Missing explicit model directory and corrupt checkpoint are rejected; no base-model fallback.
- Empty input and input exceeding 20,000 characters are rejected.
- Long input is truncated to 512 tokens with an explicit flag.
- Greedy generation and beam search work; repeated deterministic inference matches.
- Streamlit AppTest passes initial rendering, empty submission, example insertion and actual generation, without UI errors.
- Dependency consistency check passes. CPU dependencies resolve for Windows and Linux x86-64/Python 3.11. Linux resolution is not a Linux runtime test.
- Git ignores the checkpoint, virtual environments and local audit outputs. Diff whitespace checks pass.

## Observed performance

The final validation process loaded the model in 13.204 seconds. Three short synthetic news examples took 2.600, 2.228 and 1.978 seconds to summarize. These are local observations, not benchmarks or cloud latency promises. A prior cold baseline including imports was substantially slower.

The test process peaked at 839.59 MiB working set while holding two model instances (direct inference plus Streamlit AppTest). This is not a measurement of a single production server or proof that the app fits a particular cloud quota.

Example output for the sample library article:

> the building will stay open until 9 p.m. on weekdays, two hours later than before. weekend hours will remain unchanged.

This shows successful inference, not verified summary quality. No ROUGE scores were reproduced.

## Tested versions

PyTorch 2.8.0+cpu; Transformers 4.53.3; Streamlit 1.39.0; SentencePiece 0.2.2; safetensors 0.8.0; huggingface-hub 0.36.2.

## Security review

Gitleaks current-file and repository-history findings were reviewed. The reported candidates were artifact hashes and a Transformers import, not credentials. No confirmed exposed credential was identified by these scans. No credential rotation is indicated by those findings. Scanning does not guarantee the absence of every possible secret.

The app uses public model downloads without an API token. Submitted articles are not logged by the application. Runtime errors expose a generic UI message and log only the exception type.

## Remaining checks

Linux/cloud build, cloud memory and cold-start behavior, and the actual public URL must still be tested. Public upload and pinned download verification are recorded below. Individual authorship is confirmed by the owner; the training notebook is unavailable. The preserved training report remains historical evidence only.

Detailed local evidence is retained outside tracked source in `.audit/t5-20260928/validation-results.json` and `validate_t5.py`.

Browser verification: after restarting the local Streamlit server, the example article produced the expected summary in the actual browser. A screenshot is saved in docs/local-summary.png. The Python server working set observed after inference was about 593 MiB on this Windows machine; cloud resource use remains untested.

## Public artifact verification — 2026-09-28

The owner uploaded the seven files to `FaizanTariq109/t5-small-news-summarizer`, revision `1cc579fc6a968c116b2358a5a80d6c45d5baba64`. An anonymous download into a separate initially empty audit cache completed. All seven SHA-256 values matched, the downloaded model loaded successfully and generated this summary:

> the library will stay open until 9 p.m. on weekdays, two hours later than before. weekend hours will remain unchanged.

The app's actual Hub-selection branch was separately exercised using a copy of its runtime module and manifest with no adjacent weights, reusing the verified cache. Selection and integrity checks passed.

The initial large-file transfer took several minutes; verification was restarted using the completed cache. The recorded 0.38-second snapshot lookup is a cache-hit measurement, not fresh download latency. No reliable cloud cold-start timing is established. Public artifact verification is now complete; Linux/cloud execution and the live URL remain pending.
