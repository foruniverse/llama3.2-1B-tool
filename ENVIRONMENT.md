# Reproducible Environment

This project is managed with `uv` and a committed `uv.lock`.

## Server Setup

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository and create the locked environment:

```bash
git clone <repo-url>
cd llama3.2-1B-tool
uv python install 3.11.14
uv sync --locked
```

Run commands through `uv`:

```bash
uv run python scripts/2_prepare_data.py
uv run python scripts/3_sft_training.py
```

Optional monitoring and notebook tools are separated from the training runtime:

```bash
uv sync --locked --group monitor
uv sync --locked --group notebook
```

## Notes

- Python is pinned to `3.11.14` in `.python-version` and `pyproject.toml`.
- Direct runtime dependencies are pinned in `pyproject.toml`; transitive dependencies are pinned in `uv.lock`.
- `requirements.txt` is only a pip fallback. Use `uv sync --locked` for server reproduction.
- The SFT script uses `flash_attention_2` when `flash-attn` is installed, and falls back to PyTorch `sdpa` otherwise.
- You can force the attention backend with `ATTN_IMPLEMENTATION=flash_attention_2` or `ATTN_IMPLEMENTATION=sdpa`.
