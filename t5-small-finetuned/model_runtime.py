"""Load the recovered T5 checkpoint and run bounded CPU summarization."""
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import hashlib
import json
import re

APP_DIR = Path(__file__).resolve().parent
MAX_INPUT_TOKENS = 512
MAX_INPUT_CHARACTERS = 20000


@dataclass
class Runtime:
    tokenizer: object
    model: object
    lock: Lock


def verify_artifacts(folder: Path) -> None:
    manifest = json.loads((APP_DIR / 'model-manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['sha256'].items():
        path = folder / name
        if not path.is_file():
            raise ValueError(f'Required model file is missing: {name}')
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise ValueError(f'Model file does not match the verified checkpoint: {name}')


def resolve_model_folder(local_dir: str = '', repo_id: str = '', revision: str = '') -> Path:
    if local_dir:
        folder = Path(local_dir).expanduser()
        if not folder.is_absolute():
            folder = APP_DIR / folder
    elif (APP_DIR / 'model.safetensors').is_file():
        folder = APP_DIR
    else:
        if not re.fullmatch(r'[\w.-]+/[\w.-]+', repo_id):
            raise ValueError('Set T5_MODEL_REPO to the published model repository.')
        if not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('Set T5_MODEL_REVISION to its full 40-character commit SHA.')
        from huggingface_hub import snapshot_download
        manifest = json.loads((APP_DIR / 'model-manifest.json').read_text(encoding='utf-8'))
        folder = Path(snapshot_download(repo_id=repo_id, revision=revision,
                                       allow_patterns=list(manifest['sha256'])))
    verify_artifacts(folder)
    return folder


def load_runtime(local_dir: str = '', repo_id: str = '', revision: str = '') -> Runtime:
    folder = resolve_model_folder(local_dir, repo_id, revision)
    import torch
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    torch.set_num_threads(2)
    tokenizer = T5Tokenizer.from_pretrained(folder, local_files_only=True, legacy=True)
    model, info = T5ForConditionalGeneration.from_pretrained(
        folder, local_files_only=True, use_safetensors=True, output_loading_info=True)
    if any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('The recovered checkpoint does not completely match its configuration.')
    model.to('cpu').eval()
    return Runtime(tokenizer, model, Lock())


def summarize(runtime: Runtime, text: str, max_length: int = 128, num_beams: int = 4):
    text = text.strip()
    if not text:
        raise ValueError('Paste an article before generating a summary.')
    if len(text) > MAX_INPUT_CHARACTERS:
        raise ValueError('Please use an article of at most 20,000 characters.')
    if not 32 <= max_length <= 128 or not 1 <= num_beams <= 4:
        raise ValueError('Choose 32–128 output tokens and 1–4 beams.')
    import torch
    with runtime.lock, torch.inference_mode():
        full_ids = runtime.tokenizer('summarize: ' + text, add_special_tokens=True, verbose=False)['input_ids']
        inputs = runtime.tokenizer('summarize: ' + text, return_tensors='pt',
                                   max_length=MAX_INPUT_TOKENS, truncation=True)
        options = {'max_length': max_length, 'num_beams': num_beams, 'do_sample': False}
        if num_beams > 1:
            options.update(length_penalty=2.0, early_stopping=True)
        output = runtime.model.generate(**inputs, **options)
        summary = runtime.tokenizer.decode(output[0], skip_special_tokens=True).strip()
    if not summary:
        raise ValueError('The model returned no summary. Try a longer English news article.')
    return {'summary': summary, 'truncated': len(full_ids) > MAX_INPUT_TOKENS,
            'input_tokens': min(len(full_ids), MAX_INPUT_TOKENS)}
