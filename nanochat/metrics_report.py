from __future__ import annotations

import dataclasses
import datetime as dt
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_GPU_CSV_RE = re.compile(r"^gpu_(?P<phase>.+)_(?P<ts>\d{8}_\d{6})\.csv$")

# Upper bound on rendered points per (file, metric, GPU) time-series in the HTML.
# This keeps the JSON payload and Plotly rendering snappy even for long runs.
MAX_POINTS_PER_SERIES = 1000


@dataclasses.dataclass(frozen=True)
class MetricsFile:
    csv_path: Path
    phase: str
    timestamp: dt.datetime  # parsed from filename, UTC-agnostic

    @property
    def key(self) -> str:
        # stable identifier for UI selection
        return f"{self.phase}__{self.timestamp.strftime('%Y%m%d_%H%M%S')}"

    @property
    def meta_path(self) -> Path:
        return self.csv_path.with_suffix(".meta.txt")


def _default_metrics_dir() -> Path:
    base_dir = os.environ.get("NANOCHAT_BASE_DIR", str(Path.home() / ".cache" / "nanochat"))
    return Path(base_dir) / "metrics"


def discover_metrics_files(metrics_dir: Optional[Path] = None) -> List[MetricsFile]:
    metrics_dir = metrics_dir or _default_metrics_dir()
    if not metrics_dir.exists():
        return []

    out: List[MetricsFile] = []
    for p in sorted(metrics_dir.glob("gpu_*.csv")):
        m = _GPU_CSV_RE.match(p.name)
        if not m:
            continue
        phase = m.group("phase")
        ts = dt.datetime.strptime(m.group("ts"), "%Y%m%d_%H%M%S")
        out.append(MetricsFile(csv_path=p, phase=phase, timestamp=ts))
    return out


def _to_float_unit(value: Any, unit_suffix: str) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s or s.lower() == "n/a":
        return None
    if unit_suffix and s.endswith(unit_suffix):
        s = s[: -len(unit_suffix)].strip()
    # Some nvidia-smi formats include commas after CSV split spacing
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        # last resort: pull the first float-like token
        m = re.search(r"[-+]?\d+(\.\d+)?", s)
        return float(m.group(0)) if m else None


def _normalize_columns(cols: Iterable[str]) -> List[str]:
    # Strip spaces; keep original names but normalized
    out = []
    for c in cols:
        out.append(c.strip())
    return out


def load_metrics_dataframe(csv_path: Path):
    """
    Load a single nvidia-smi CSV (as written by our scripts) into a DataFrame with:
      - ts: datetime
      - gpu_index: int
      - gpu_util_pct, mem_util_pct: float
      - power_w: float
      - sm_clock_mhz, mem_clock_mhz: float
      - temp_c: float
      - mem_used_mib, mem_total_mib: float
    """
    import pandas as pd

    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.columns = _normalize_columns(df.columns)

    # Find columns robustly (nvidia-smi adds units to headers)
    def find_col(prefix: str) -> str:
        for c in df.columns:
            if c.startswith(prefix):
                return c
        raise KeyError(f"Missing expected column starting with {prefix!r} in {csv_path}")

    col_ts = find_col("timestamp")
    col_index = find_col("index")
    col_gpu_util = find_col("utilization.gpu")
    col_mem_util = find_col("utilization.memory")
    col_power = find_col("power.draw")
    col_sm_clock = find_col("clocks.current.sm")
    col_mem_clock = find_col("clocks.current.memory")
    col_temp = find_col("temperature.gpu")
    col_mem_used = find_col("memory.used")
    col_mem_total = find_col("memory.total")

    # Parse timestamp (format observed: YYYY/MM/DD HH:MM:SS.mmm)
    df["ts"] = pd.to_datetime(df[col_ts], errors="coerce")
    df["gpu_index"] = pd.to_numeric(df[col_index], errors="coerce").astype("Int64")

    df["gpu_util_pct"] = df[col_gpu_util].map(lambda v: _to_float_unit(v, "%"))
    df["mem_util_pct"] = df[col_mem_util].map(lambda v: _to_float_unit(v, "%"))
    df["power_w"] = df[col_power].map(lambda v: _to_float_unit(v, "W"))
    df["sm_clock_mhz"] = df[col_sm_clock].map(lambda v: _to_float_unit(v, "MHz"))
    df["mem_clock_mhz"] = df[col_mem_clock].map(lambda v: _to_float_unit(v, "MHz"))
    df["temp_c"] = df[col_temp].map(lambda v: _to_float_unit(v, ""))  # unitless integer
    df["mem_used_mib"] = df[col_mem_used].map(lambda v: _to_float_unit(v, "MiB"))
    df["mem_total_mib"] = df[col_mem_total].map(lambda v: _to_float_unit(v, "MiB"))

    # Drop rows where timestamp or index couldn't parse
    df = df.dropna(subset=["ts", "gpu_index"])
    df["gpu_index"] = df["gpu_index"].astype(int)

    return df


def parse_meta_file(meta_path: Path) -> Dict[str, Any]:
    """
    Best-effort parse of our .meta.txt file format.
    Returns keys like: phase, started_at, interval_seconds, gpus(list), driver_version, pci_bus_id.
    """
    if not meta_path.exists():
        return {}
    text = meta_path.read_text(encoding="utf-8", errors="replace")
    out: Dict[str, Any] = {}
    for line in text.splitlines():
        if "=" in line and not line.startswith("["):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    # Parse GPU list
    gpus: List[Dict[str, str]] = []
    for line in text.splitlines():
        if line.startswith("GPU ") and ":" in line and "(UUID:" in line:
            # GPU 0: NVIDIA RTX A5000 (UUID: ...)
            m = re.match(r"GPU\s+(?P<idx>\d+):\s+(?P<name>.+?)\s+\(UUID:\s+(?P<uuid>.+)\)", line)
            if m:
                gpus.append({"index": m.group("idx"), "name": m.group("name"), "uuid": m.group("uuid")})
    if gpus:
        out["gpus"] = gpus
    return out


def build_metrics_index(metrics_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Returns a JSON-serializable structure describing all discovered metrics files,
    including quick summary stats per file.
    """
    metrics_dir = metrics_dir or _default_metrics_dir()
    files = discover_metrics_files(metrics_dir)

    index: Dict[str, Any] = {"metrics_dir": str(metrics_dir), "files": []}
    for f in files:
        meta = parse_meta_file(f.meta_path)
        index["files"].append(
            {
                "key": f.key,
                "phase": f.phase,
                "timestamp": f.timestamp.strftime("%Y%m%d_%H%M%S"),
                "csv_path": str(f.csv_path),
                "meta": meta,
            }
        )
    return index


def generate_html_report(metrics_dir: Optional[Path] = None, out_path: Optional[Path] = None) -> Path:
    """
    Build a self-contained HTML report with selectors (phase/run via file key)
    and interactive Plotly charts.
    """
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.io as pio

    metrics_dir = metrics_dir or _default_metrics_dir()
    out_path = out_path or (metrics_dir / "report.html")

    files = discover_metrics_files(metrics_dir)
    if not files:
        raise FileNotFoundError(f"No gpu_*.csv files found under {metrics_dir}")

    # Build figure JSON per (file, metric)
    metric_specs = [
        ("gpu_util_pct", "GPU Utilization (%)", "Percent"),
        ("mem_used_mib", "GPU Memory Used (MiB)", "MiB"),
        ("power_w", "Power Draw (W)", "W"),
        ("temp_c", "Temperature (C)", "C"),
        ("sm_clock_mhz", "SM Clock (MHz)", "MHz"),
        ("mem_clock_mhz", "Memory Clock (MHz)", "MHz"),
    ]

    figures: Dict[str, Dict[str, Any]] = {}
    summaries: Dict[str, Any] = {}
    meta_by_key: Dict[str, Any] = {}

    for mf in files:
        df = load_metrics_dataframe(mf.csv_path)
        meta_by_key[mf.key] = parse_meta_file(mf.meta_path)

        # Summary per file (across all GPUs)
        summaries[mf.key] = {
            "time_start": df["ts"].min().isoformat() if len(df) else None,
            "time_end": df["ts"].max().isoformat() if len(df) else None,
            "gpu_count": int(df["gpu_index"].nunique()) if len(df) else 0,
            "rows": int(len(df)),
        }

        for metric, title, yunit in metric_specs:
            fig = go.Figure()
            for gpu_idx in sorted(df["gpu_index"].unique()):
                sdf = df[df["gpu_index"] == gpu_idx]
                # Downsample long, mostly smooth series to keep the HTML light.
                n = len(sdf)
                if n > MAX_POINTS_PER_SERIES:
                    step = max(1, math.ceil(n / MAX_POINTS_PER_SERIES))
                    sdf = sdf.iloc[::step]
                fig.add_trace(
                    go.Scatter(
                        x=sdf["ts"],
                        y=sdf[metric],
                        mode="lines",
                        name=f"GPU {gpu_idx}",
                    )
                )
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title=yunit,
                legend_title="GPU",
                margin=dict(l=40, r=20, t=60, b=40),
                height=520,
            )
            # Serialize via Plotly's JSON engine to avoid np.ndarray objects
            fig_key = f"{mf.key}__{metric}"
            figures[fig_key] = pio.to_json(fig, validate=False, engine="json")

    # Simple run grouping: by date (YYYYmmdd) inferred from filename timestamp
    runs: Dict[str, List[str]] = {}
    for mf in files:
        run_id = mf.timestamp.strftime("%Y%m%d")
        runs.setdefault(run_id, []).append(mf.key)
    # Ensure stable ordering
    runs = {k: sorted(v) for k, v in sorted(runs.items())}

    payload = {
        "metrics_dir": str(metrics_dir),
        "runs": runs,
        "files": [
            {
                "key": mf.key,
                "phase": mf.phase,
                "timestamp": mf.timestamp.strftime("%Y%m%d_%H%M%S"),
            }
            for mf in files
        ],
        "metric_specs": [{"key": m, "title": t, "unit": u} for (m, t, u) in metric_specs],
        "summaries": summaries,
        "meta_by_key": meta_by_key,
        "figures": figures,
        "generated_at": dt.datetime.now().isoformat(),
    }

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>nanochat GPU Metrics Report</title>
  <style>
    body {{ font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin: 0; }}
    header {{ padding: 14px 18px; border-bottom: 1px solid #e5e7eb; }}
    .container {{ display: grid; grid-template-columns: 360px 1fr; min-height: calc(100vh - 54px); }}
    .sidebar {{ border-right: 1px solid #e5e7eb; padding: 14px 18px; }}
    .content {{ padding: 14px 18px; }}
    label {{ display:block; font-size: 12px; margin-top: 12px; color:#374151; }}
    select {{ width: 100%; padding: 8px; margin-top: 6px; }}
    .kpi {{ display:grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; margin-top: 12px; }}
    .card {{ border: 1px solid #e5e7eb; border-radius: 10px; padding: 10px; }}
    .card .k {{ font-size: 12px; color:#6b7280; }}
    .card .v {{ font-size: 14px; font-weight: 600; margin-top: 2px; }}
    .meta {{ margin-top: 14px; font-size: 12px; color:#374151; white-space: pre-wrap; }}
  </style>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
</head>
<body>
  <header>
    <div style="font-weight:700">nanochat GPU Metrics Report</div>
    <div style="font-size:12px;color:#6b7280">Generated at: {payload["generated_at"]} | Source: {payload["metrics_dir"]}</div>
  </header>
  <div class="container">
    <div class="sidebar">
      <label>Run</label>
      <select id="runSel"></select>
      <label>Phase instance</label>
      <select id="fileSel"></select>
      <label>Metric</label>
      <select id="metricSel"></select>

      <div class="kpi" id="kpi"></div>
      <div class="meta" id="meta"></div>
    </div>
    <div class="content">
      <div id="plot"></div>
    </div>
  </div>

  <script>
  const DATA = {json.dumps(payload)};

  function el(id) {{ return document.getElementById(id); }}

  function fillSelect(selectEl, options, selectedValue) {{
    selectEl.innerHTML = '';
    for (const opt of options) {{
      const o = document.createElement('option');
      o.value = opt.value;
      o.textContent = opt.label;
      if (selectedValue && opt.value === selectedValue) o.selected = true;
      selectEl.appendChild(o);
    }}
  }}

  function updateFilesForRun(runId) {{
    const keys = DATA.runs[runId] || [];
    const options = keys.map(k => {{
      const f = DATA.files.find(x => x.key === k);
      const label = f ? `${{f.timestamp}} — ${{f.phase}}` : k;
      return {{ value: k, label }};
    }});
    fillSelect(el('fileSel'), options, options[0]?.value);
  }}

  function updateKpiAndMeta(fileKey) {{
    const s = DATA.summaries[fileKey] || {{}};
    const cards = [
      {{k:'Start', v: s.time_start || '-'}},
      {{k:'End', v: s.time_end || '-'}},
      {{k:'GPUs', v: (s.gpu_count ?? '-')}},
      {{k:'Rows', v: (s.rows ?? '-')}},
    ];
    el('kpi').innerHTML = '';
    for (const c of cards) {{
      const div = document.createElement('div');
      div.className = 'card';
      div.innerHTML = `<div class="k">${{c.k}}</div><div class="v">${{c.v}}</div>`;
      el('kpi').appendChild(div);
    }}

    const meta = DATA.meta_by_key[fileKey] || {{}};
    el('meta').textContent = Object.keys(meta).length ? JSON.stringify(meta, null, 2) : '';
  }}

  function renderPlot(fileKey, metricKey) {{
    const figKey = `${{fileKey}}__${{metricKey}}`;
    const raw = DATA.figures[figKey];
    const fig = raw ? JSON.parse(raw) : null;
    if (!fig) {{
      Plotly.purge('plot');
      el('plot').innerHTML = '<div style=\"color:#ef4444\">Missing figure: ' + figKey + '</div>';
      return;
    }}
    Plotly.react('plot', fig.data, fig.layout, {{responsive: true}});
  }}

  // init selectors
  const runOptions = Object.keys(DATA.runs).map(r => ({{value:r, label:r}}));
  fillSelect(el('runSel'), runOptions, runOptions[runOptions.length - 1]?.value);

  const metricOptions = DATA.metric_specs.map(m => ({{value:m.key, label:m.title}}));
  fillSelect(el('metricSel'), metricOptions, metricOptions[0]?.value);

  // wire events
  el('runSel').addEventListener('change', () => {{
    updateFilesForRun(el('runSel').value);
    const fileKey = el('fileSel').value;
    updateKpiAndMeta(fileKey);
    renderPlot(fileKey, el('metricSel').value);
  }});
  el('fileSel').addEventListener('change', () => {{
    const fileKey = el('fileSel').value;
    updateKpiAndMeta(fileKey);
    renderPlot(fileKey, el('metricSel').value);
  }});
  el('metricSel').addEventListener('change', () => {{
    renderPlot(el('fileSel').value, el('metricSel').value);
  }});

  // first render
  updateFilesForRun(el('runSel').value);
  updateKpiAndMeta(el('fileSel').value);
  renderPlot(el('fileSel').value, el('metricSel').value);
  </script>
</body>
</html>
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path

