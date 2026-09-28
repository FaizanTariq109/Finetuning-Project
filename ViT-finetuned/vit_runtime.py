"""CPU inference for the recovered ViT checkpoint; no training or remote code."""
import hashlib
import io
import json
import threading
import warnings
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from PIL import Image, ImageOps, UnidentifiedImageError
from transformers import ViTConfig, ViTForImageClassification, ViTImageProcessor

BASE_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 16_000_000
_INFERENCE_LOCK = threading.Lock()


def read_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Choose a JPG or PNG image no larger than 10 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Choose a valid JPG or PNG image.")
                if image.width * image.height > MAX_PIXELS:
                    raise ValueError("Choose an image with at most 16 million pixels.")
                image.load()
                return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ValueError("This image could not be read. Try another JPG or PNG.") from exc


def verify_file(path, expected):
    path = Path(path)
    if path.suffix == '.json':
        digest = hashlib.sha256(json.dumps(json.loads(path.read_text()), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if digest != expected['canonical_sha256']:
            raise ValueError('Model configuration checksum mismatch.')
        return
    if path.stat().st_size != expected['size']:
        raise ValueError("Model artifact size mismatch.")
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected['sha256']:
        raise ValueError("Model artifact checksum mismatch.")


def load_model():
    manifest = json.loads((BASE_DIR / 'artifact-manifest.json').read_text())
    for name in ['config.json', 'preprocessor_config.json']:
        verify_file(BASE_DIR / name, manifest['files'][name])
    # A local recovered weight takes precedence. Otherwise use the exact public snapshot.
    local = BASE_DIR / 'model.safetensors'
    weight = local if local.exists() else Path(hf_hub_download(
        repo_id=manifest['repo_id'], filename='model.safetensors',
        revision=manifest['revision'], token=False,
    ))
    verify_file(weight, manifest['files']['model.safetensors'])
    torch.set_num_threads(2)
    config = ViTConfig.from_pretrained(BASE_DIR, local_files_only=True)
    processor = ViTImageProcessor.from_pretrained(BASE_DIR, local_files_only=True)
    model, info = ViTForImageClassification.from_pretrained(
        weight.parent, config=config, local_files_only=True,
        use_safetensors=True, output_loading_info=True,
    )
    if any(info.values()):
        raise ValueError("Checkpoint does not match its saved configuration.")
    if model.classifier.out_features != 101 or set(config.id2label) != set(range(101)):
        raise ValueError("Unexpected classifier labels.")
    model.eval()
    return processor, model


def predict(image, processor, model):
    with _INFERENCE_LOCK, torch.inference_mode():
        inputs = processor(images=image.convert('RGB'), return_tensors='pt')
        probabilities = model(**inputs).logits.softmax(dim=-1)[0]
        if probabilities.shape != (101,) or not torch.isfinite(probabilities).all():
            raise ValueError("Invalid model output.")
        values, indices = probabilities.topk(5)
    return [{'id': i, 'label': model.config.id2label[i], 'probability': value}
            for i, value in zip(indices.tolist(), values.tolist())]
