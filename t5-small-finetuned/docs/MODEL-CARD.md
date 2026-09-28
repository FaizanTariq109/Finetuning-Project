---
language: en
library_name: transformers
pipeline_tag: summarization
tags:
- t5
- text2text-generation
- summarization
- academic-project
---
# T5-small News Summarizer

A recovered T5-small checkpoint from Faizan Tariq's individual university fine-tuning project. Intended for generating short summaries of English news articles.

## Provenance

The original project report describes fine-tuning T5-small on CNN/DailyMail v3.0.0 for two epochs. The training notebook is unavailable. Training details and historical ROUGE results have not been independently reproduced, so this card does not advertise benchmark scores.

Faizan Tariq completed the fine-tuning and Streamlit application individually. The underlying T5 architecture and pretrained model are the work of the original T5 authors.

## Verified properties

- T5 encoder-decoder model with 60,506,624 parameters.
- Recovered configuration, tokenizer and safetensors checkpoint load together without missing, unexpected or mismatched model keys.
- CPU inference and the local Streamlit UI produce summaries on synthetic news examples.
- Model quality has not been established by these smoke tests.

Checkpoint SHA-256: `881ecd3d2e20cfb1ddf96ac598ad90a610007dfa4ff04cfc024e4afd118bc686`.

## Usage

Tested with Python 3.11, PyTorch 2.8.0+cpu, Transformers 4.53.3, SentencePiece 0.2.2 and safetensors 0.8.0.

```python
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

repo = 'FaizanTariq109/t5-small-news-summarizer'
# This immutable revision contains the verified seven-file model bundle.
revision = '1cc579fc6a968c116b2358a5a80d6c45d5baba64'
tokenizer = T5Tokenizer.from_pretrained(repo, revision=revision, legacy=True)
model = T5ForConditionalGeneration.from_pretrained(
    repo, revision=revision, use_safetensors=True
).eval()

article = 'Paste an English news article here.'
inputs = tokenizer('summarize: ' + article, return_tensors='pt',
                   max_length=512, truncation=True)
with torch.inference_mode():
    output = model.generate(**inputs, max_length=128, num_beams=4,
                            length_penalty=2.0, early_stopping=True,
                            do_sample=False)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Limitations

Only the first 512 input tokens are processed by this example. Summaries may omit important details or introduce errors; verify against the source. Non-English text, other domains and high-stakes uses have not been validated. No document chunking or factuality guarantee is provided.

## Application and attribution

Application repository: https://github.com/FaizanTariq109/Finetuning-Project (prepared T5 updates are pending publication at the time this card was drafted). A public Streamlit deployment has not yet been verified.

T5: https://huggingface.co/google-t5/t5-small

T5 paper: https://arxiv.org/abs/1910.10683

CNN/DailyMail dataset described by the historical report: https://huggingface.co/datasets/abisee/cnn_dailymail

A license for these recovered fine-tuned weights has not been specified by the owner. This card does not grant additional rights to third-party models or dataset content.
