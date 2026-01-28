# nanochat instructions

## Run the tinyrun metrics pipeline in `screen`

If you’re going to kick off a long training run and come back later, use `screen`.

### One-time setup (before starting `screen`)

Make sure `uv` and (optionally) Rust/Cargo are on your `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
export PATH="$HOME/.cargo/bin:$PATH"
```

If `uv` isn’t installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

Sync dependencies (includes `pandas` + `plotly` for the HTML GPU metrics report):

```bash
cd ~/nanochat
~/.local/bin/uv sync --extra gpu
```

### Start the run in `screen`

```bash
cd ~/nanochat
screen -S tinyrun_metrics
./tinyrun_metrics.sh
```

- Detach: `Ctrl+A` then `D`
- Reattach later:

```bash
screen -r tinyrun_metrics
```

## Generate a single HTML GPU metrics report (after the run)

This scans `nvidia-smi` CSVs under `$NANOCHAT_BASE_DIR/metrics` (default: `~/.cache/nanochat/metrics`) and produces one self-contained HTML file you can open locally.

```bash
cd ~/nanochat
python -m scripts.metrics_report --metrics-dir "$HOME/.cache/nanochat/metrics" --out "$HOME/.cache/nanochat/metrics/report.html"
```

Open:

- `~/.cache/nanochat/metrics/report.html`

