from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "google/owlv2-base-patch16-ensemble"
MODEL_REVISION = "cfd3195ba4ea9592eec887ded089f4c08eff231d"
MODEL_DIR = Path(__file__).resolve().parent / "checkpoints" / "owlv2-base-patch16-ensemble"

# Pin the Hub commit and download only inference files. The weight file is ignored by Git.
snapshot_download(
    repo_id=MODEL_ID,
    revision=MODEL_REVISION,
    local_dir=MODEL_DIR,
    allow_patterns=[
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "special_tokens_map.json",
        "added_tokens.json",
    ],
)

print("Model:", MODEL_ID)
print("Revision:", MODEL_REVISION)
print("Local files:", MODEL_DIR)
