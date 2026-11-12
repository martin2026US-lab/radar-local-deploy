"""Verified inference-only checkpoint, keeping all trained text and vision tensors."""
from pathlib import Path
import hashlib,json
import os
ROOT=Path(os.environ['RADAR_ASSET_ROOT']).resolve()
ORIGINAL_SHA='5e8b1b50b92162fcafcf8231e51ffbe9d1be2be52ccff4f967e4beb71e4499f4'
COMPACT=ROOT/'models/radar_inference_only_v1.pth'
MANIFEST=ROOT/'models/radar_inference_only_v1.manifest.json'

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def verified_checkpoint():
    manifest=json.loads(MANIFEST.read_text(encoding='utf-8'))
    if manifest['original_sha256']!=ORIGINAL_SHA or not manifest['all_model_tensors_exactly_equal']:
        raise ValueError('Invalid compact model provenance')
    if file_sha(COMPACT)!=manifest['compact_sha256']:
        raise ValueError('Compact checkpoint hash mismatch')
    return COMPACT,manifest

def text_encoder_from_config(encoder_class):
    # Upstream XBertEncoder builds its MLM head after BertModel.__init__.
    # Defer initialization until that head exists, as from_pretrained does.
    # Every active tensor will then be overwritten by the RADAR checkpoint.
    from transformers.modeling_utils import no_init_weights
    with no_init_weights():
        encoder=encoder_class.from_config({},from_pretrained=False)
    encoder.init_weights()
    return encoder
