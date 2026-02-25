"""
Dataset Statistics Generator
=============================

Generates interactive HTML charts from the merge-pipeline output.

Reads:
    registries/anotaciones.xlsx  — main annotation registry
    Duplicados/duplicados_registro.xlsx  — duplicate records

Columns used from anotaciones.xlsx:
    uuid, specimen_id, original_path, current_path,
    macroclass_label, class_label, genera_label,
    campaign_year, fuente, comentarios, created_at

Produces a single ``stats_report.html`` with all figures embedded,
plus individual PNGs in a ``figures/`` subfolder.
"""

import re
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# ── Palette ───────────────────────────────────────────────────────
PALETTE = px.colors.qualitative.Bold
BG_COLOR = "#0e1117"
CARD_BG = "#1a1f2e"
TEXT_COLOR = "#e0e0e0"
GRID_COLOR = "#2a3040"
ACCENT = "#0078d4"

_LAYOUT_DEFAULTS = dict(
    paper_bgcolor=BG_COLOR,
    plot_bgcolor=CARD_BG,
    font=dict(family="Segoe UI, sans-serif", color=TEXT_COLOR, size=13),
    margin=dict(l=60, r=30, t=60, b=50),
)


def _apply_layout(fig: go.Figure, title: str, **extra):
    """Apply dark-theme defaults to a figure.  Never touches axes."""
    merged = {**_LAYOUT_DEFAULTS, **extra}
    fig.update_layout(title=dict(text=title, font=dict(size=18)),
                      **merged)
    # Apply grid colour to all axes unless caller styled them already
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


# ── Helpers ───────────────────────────────────────────────────────

def _safe(val) -> str:
    """Convert value to clean string; NaN/None → ''."""
    if val is None:
        return ""
    s = str(val).strip()
    return "" if s.lower() == "nan" else s


_NUMERIC_RE = re.compile(r'\d+')


def _safe_year(val) -> str:
    """Normalise a campaign_year value to a clean '2009' string."""
    s = _safe(val)
    if not s:
        return ""
    # Handle floats like '2009.0'
    try:
        n = int(float(s))
        if 1800 <= n <= 2100:
            return str(n)
    except (ValueError, OverflowError):
        pass
    return s


def _extract_numeric_specimen(sid: str) -> str:
    """Extract just the numeric part of a specimen ID for grouping."""
    nums = _NUMERIC_RE.findall(sid)
    return "".join(nums) if nums else sid


# ═════════════════════════════════════════════════════════════════
# Loading
# ═════════════════════════════════════════════════════════════════

def load_data(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load anotaciones + duplicados from the merge output directory."""
    reg = None
    for name in ("registries", "registros"):
        p = output_dir / name
        if p.exists():
            reg = p
            break
    if reg is None:
        sys.exit("ERROR: No 'registries/' or 'registros/' directory found.")

    ann_path = reg / "anotaciones.xlsx"
    if not ann_path.exists():
        sys.exit(f"ERROR: {ann_path} not found.")

    ann = pd.read_excel(ann_path)
    logger.info(f"Loaded {len(ann)} annotations from {ann_path}")

    dup_path = output_dir / "Duplicados" / "duplicados_registro.xlsx"
    if dup_path.exists():
        dup = pd.read_excel(dup_path)
        logger.info(f"Loaded {len(dup)} duplicate records from {dup_path}")
    else:
        dup = pd.DataFrame()
        logger.warning("duplicados_registro.xlsx not found — duplicate stats will be skipped.")

    return ann, dup


# ═════════════════════════════════════════════════════════════════
# 1. Duplicates vs Non-Duplicates
# ═════════════════════════════════════════════════════════════════

def fig_duplicates_ratio(ann: pd.DataFrame, dup: pd.DataFrame) -> go.Figure:
    """Donut chart + headline numbers: unique images vs duplicates."""
    n_main = len(ann)
    n_dup = len(dup) if len(dup) > 0 else 0
    n_total = n_main + n_dup

    labels = ["Unique images", "Duplicates"]
    values = [n_main, n_dup]
    colors = ["#3fb950", "#f85149"]

    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=BG_COLOR, width=2)),
        textinfo="label+value+percent",
        textfont=dict(size=14),
        hovertemplate="%{label}: %{value:,}<br>%{percent}<extra></extra>",
    ))

    ratio = n_dup / n_main if n_main else 0
    fig.add_annotation(
        text=f"<b>{ratio:.2f}</b><br><span style='font-size:11px'>dup / img</span>",
        showarrow=False, font=dict(size=22, color=TEXT_COLOR), x=0.5, y=0.5,
    )
    _apply_layout(fig, f"Duplicates vs Unique Images  (total files: {n_total:,})",
                  showlegend=True, legend=dict(orientation="h", y=-0.1))
    return fig


# ═════════════════════════════════════════════════════════════════
# 2. Distribution per Year
# ═════════════════════════════════════════════════════════════════

def fig_year_distribution(ann: pd.DataFrame) -> go.Figure:
    """Vertical bar chart of images per campaign year."""
    years = ann["campaign_year"].apply(_safe_year).replace("", "Unknown")
    counts = years.value_counts()

    # Sort numerically
    numeric_keys = []
    other_keys = []
    for k in counts.index:
        try:
            numeric_keys.append((int(k), k))
        except ValueError:
            other_keys.append(k)
    numeric_keys.sort()
    order = [k for _, k in numeric_keys] + sorted(other_keys)
    counts = counts.reindex(order)

    x_labels = [str(v) for v in counts.index]
    y_values = [int(v) for v in counts.values]
    colors = [ACCENT if y != "Unknown" else "#555555" for y in x_labels]

    fig = go.Figure(go.Bar(
        x=x_labels,
        y=y_values,
        orientation="v",
        marker_color=colors,
        text=y_values,
        textposition="outside",
        hovertemplate="Year %{x}: %{y:,} images<extra></extra>",
    ))
    fig.update_xaxes(type="category", title_text="Year")
    fig.update_yaxes(title_text="Image count")
    _apply_layout(fig, "Images per Campaign Year")
    return fig


# ═════════════════════════════════════════════════════════════════
# 3. Distribution per Macroclass
# ═════════════════════════════════════════════════════════════════

def fig_macroclass_distribution(ann: pd.DataFrame) -> go.Figure:
    """Vertical bar chart of images per macroclass."""
    mc = ann["macroclass_label"].apply(_safe).replace("", "Unknown")
    counts = mc.value_counts().sort_values(ascending=False)

    x_labels = list(counts.index)
    y_values = [int(v) for v in counts.values]
    n_cats = len(x_labels)
    colors = [PALETTE[i % len(PALETTE)] for i in range(n_cats)]

    fig = go.Figure(go.Bar(
        x=x_labels,
        y=y_values,
        orientation="v",
        marker_color=colors,
        text=y_values,
        textposition="outside",
        hovertemplate="%{x}: %{y:,} images<extra></extra>",
    ))
    fig.update_xaxes(type="category", tickangle=-30, title_text="")
    fig.update_yaxes(title_text="Image count")
    _apply_layout(fig, "Images per Macroclass", height=500)
    return fig


# ═════════════════════════════════════════════════════════════════
# 4. Macroclass → Class → Genera  (Sunburst + Sankey)
# ═════════════════════════════════════════════════════════════════

def fig_taxonomy_sunburst(ann: pd.DataFrame) -> go.Figure:
    """Sunburst: Macroclass → Class → Genera hierarchy."""
    df = ann[["macroclass_label", "class_label", "genera_label"]].copy()
    df["macroclass_label"] = df["macroclass_label"].apply(_safe).replace("", "Unknown")
    df["class_label"] = df["class_label"].apply(_safe).replace("", "Unknown")
    df["genera_label"] = df["genera_label"].apply(_safe).replace("", "Unknown")

    fig = px.sunburst(
        df, path=["macroclass_label", "class_label", "genera_label"],
        color="macroclass_label",
        color_discrete_sequence=PALETTE,
    )
    _apply_layout(fig, "Taxonomy Hierarchy  (Macroclass → Class → Genera)",
                  height=700)
    fig.update_traces(
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Parent: %{parent}<extra></extra>",
    )
    return fig


def fig_taxonomy_treemap(ann: pd.DataFrame) -> go.Figure:
    """Treemap: Macroclass → Class → Genera hierarchy."""
    df = ann[["macroclass_label", "class_label", "genera_label"]].copy()
    df["macroclass_label"] = df["macroclass_label"].apply(_safe).replace("", "Unknown")
    df["class_label"] = df["class_label"].apply(_safe).replace("", "Unknown")
    df["genera_label"] = df["genera_label"].apply(_safe).replace("", "Unknown")

    fig = px.treemap(
        df, path=["macroclass_label", "class_label", "genera_label"],
        color="macroclass_label",
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(
        textinfo="label+value",
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Parent: %{parent}<extra></extra>",
    )
    _apply_layout(fig, "Taxonomy Treemap  (Macroclass → Class → Genera)",
                  height=700)
    return fig


def fig_taxonomy_sankey(ann: pd.DataFrame) -> go.Figure:
    """Sankey diagram flowing from Macroclass → Class → Genera."""
    df = ann[["macroclass_label", "class_label", "genera_label"]].copy()
    df["macroclass_label"] = df["macroclass_label"].apply(_safe).replace("", "Unknown")
    df["class_label"] = df["class_label"].apply(_safe).replace("", "Unknown")
    df["genera_label"] = df["genera_label"].apply(_safe).replace("", "Unknown")

    # Make "Unknown" unique per parent so they don't merge into a single node
    df.loc[df["class_label"] == "Unknown", "class_label"] = (
        "Unknown (" + df.loc[df["class_label"] == "Unknown", "macroclass_label"] + ")"
    )
    df.loc[df["genera_label"] == "Unknown", "genera_label"] = (
        "Unknown (" + df.loc[df["genera_label"] == "Unknown", "class_label"] + ")"
    )

    # Build nodes and links for the three levels
    # Limit genera to top-N to keep the chart readable
    TOP_GENERA = 25

    # -- Macroclass → Class links --
    mc_cls = df.groupby(["macroclass_label", "class_label"]).size().reset_index(name="count")

    # -- Class → Genera links (top genera only) --
    cls_gen = df.groupby(["class_label", "genera_label"]).size().reset_index(name="count")
    top_genera = cls_gen.groupby("genera_label")["count"].sum().nlargest(TOP_GENERA).index
    cls_gen = cls_gen[cls_gen["genera_label"].isin(top_genera)]

    # Collect all unique labels, prefixed to avoid collisions
    macro_labels = sorted(mc_cls["macroclass_label"].unique())
    class_labels = sorted(set(mc_cls["class_label"].unique()) | set(cls_gen["class_label"].unique()))
    genera_labels = sorted(cls_gen["genera_label"].unique())

    node_labels = (
        [f"M: {l}" for l in macro_labels] +
        [f"C: {l}" for l in class_labels] +
        [f"G: {l}" for l in genera_labels]
    )

    idx = {label: i for i, label in enumerate(node_labels)}

    source, target, value = [], [], []

    for _, row in mc_cls.iterrows():
        s = idx.get(f"M: {row['macroclass_label']}")
        t = idx.get(f"C: {row['class_label']}")
        if s is not None and t is not None:
            source.append(s); target.append(t); value.append(row["count"])

    for _, row in cls_gen.iterrows():
        s = idx.get(f"C: {row['class_label']}")
        t = idx.get(f"G: {row['genera_label']}")
        if s is not None and t is not None:
            source.append(s); target.append(t); value.append(row["count"])

    # Colors per node group
    n_m = len(macro_labels)
    n_c = len(class_labels)
    n_g = len(genera_labels)
    node_colors = (
        [PALETTE[i % len(PALETTE)] for i in range(n_m)] +
        ["#4a6785"] * n_c +
        ["#6a7a8a"] * n_g
    )

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18, thickness=20, line=dict(color=BG_COLOR, width=1),
            label=[l.split(": ", 1)[-1] for l in node_labels],
            color=node_colors,
        ),
        link=dict(
            source=source, target=target, value=value,
            color="rgba(100,150,200,0.25)",
        ),
    ))
    subtitle = f"  (showing top {TOP_GENERA} genera)" if len(genera_labels) > 0 else ""
    _apply_layout(fig, f"Taxonomy Flow: Macroclass → Class → Genera{subtitle}",
                  height=700)
    return fig


# ═════════════════════════════════════════════════════════════════
# 5. Specimen ID Presence
# ═════════════════════════════════════════════════════════════════

def fig_specimen_id_presence(ann: pd.DataFrame) -> go.Figure:
    """Pie chart: images with vs without specimen_id."""
    has_id = ann["specimen_id"].apply(_safe).ne("").sum()
    no_id = len(ann) - has_id

    fig = go.Figure(go.Pie(
        labels=["With Specimen ID", "Without Specimen ID"],
        values=[has_id, no_id],
        hole=0.5,
        marker=dict(colors=["#58a6ff", "#f85149"],
                    line=dict(color=BG_COLOR, width=2)),
        textinfo="label+value+percent",
        textfont=dict(size=14),
    ))
    _apply_layout(fig, "Images With vs Without Specimen ID",
                  showlegend=True, legend=dict(orientation="h", y=-0.1))
    fig.add_annotation(
        text=f"<b>{has_id / len(ann) * 100:.1f}%</b><br><span style='font-size:11px'>identified</span>",
        showarrow=False, font=dict(size=20, color=TEXT_COLOR), x=0.5, y=0.5,
    )
    return fig


# ═════════════════════════════════════════════════════════════════
# 6. Shared Specimen IDs (numeric part)
# ═════════════════════════════════════════════════════════════════

def fig_shared_specimen_ids(ann: pd.DataFrame) -> go.Figure:
    """Donut showing how many unique specimen IDs appear on >1 image."""
    sids = ann["specimen_id"].apply(_safe)
    sids = sids[sids != ""]
    numeric = sids.apply(_extract_numeric_specimen)
    counts = numeric.value_counts()

    n_unique = (counts == 1).sum()
    n_shared = (counts > 1).sum()
    total_ids = n_unique + n_shared

    fig = go.Figure(go.Pie(
        labels=["Unique (1 image)", "Shared (>1 image)"],
        values=[n_unique, n_shared],
        hole=0.55,
        marker=dict(colors=["#58a6ff", "#e8853d"],
                    line=dict(color=BG_COLOR, width=2)),
        textinfo="label+value+percent",
        textfont=dict(size=14),
        hovertemplate="%{label}: %{value:,} specimen IDs<extra></extra>",
    ))
    fig.add_annotation(
        text=f"<b>{n_shared:,}</b><br><span style='font-size:11px'>shared IDs</span>",
        showarrow=False, font=dict(size=22, color=TEXT_COLOR), x=0.5, y=0.5,
    )
    _apply_layout(fig,
                  f"Specimen IDs on Multiple Images  ({total_ids:,} total unique IDs)",
                  showlegend=True, legend=dict(orientation="h", y=-0.1))
    return fig


# ═════════════════════════════════════════════════════════════════
# 7. BONUS: Field Completeness
# ═════════════════════════════════════════════════════════════════

def fig_field_completeness(ann: pd.DataFrame) -> go.Figure:
    """Stacked bar showing filled vs empty for each annotation field."""
    fields = ["specimen_id", "macroclass_label", "class_label",
              "genera_label", "campaign_year", "fuente", "comentarios"]

    available = [f for f in fields if f in ann.columns]
    filled_counts = []
    empty_counts = []
    labels = []

    for col in available:
        vals = ann[col].apply(_safe)
        n_filled = (vals != "").sum()
        n_empty = len(ann) - n_filled
        filled_counts.append(n_filled)
        empty_counts.append(n_empty)
        labels.append(col.replace("_", " ").title())

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=filled_counts, orientation="h",
        name="Filled", marker_color="#3fb950",
        text=filled_counts, textposition="inside",
        hovertemplate="%{y}: %{x:,} filled<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        y=labels, x=empty_counts, orientation="h",
        name="Empty", marker_color="rgba(248,81,73,0.38)",
        text=empty_counts, textposition="inside",
        hovertemplate="%{y}: %{x:,} empty<extra></extra>",
    ))

    _apply_layout(fig, "Annotation Field Completeness",
                  barmode="stack",
                  showlegend=True, legend=dict(orientation="h", y=-0.12),
                  height=max(350, 45 * len(available)))
    fig.update_xaxes(title_text="Image count")
    fig.update_yaxes(title_text="")
    return fig


# ═════════════════════════════════════════════════════════════════
# 9. BONUS: Macroclass × Year Heatmap
# ═════════════════════════════════════════════════════════════════

def fig_macroclass_year_heatmap(ann: pd.DataFrame) -> go.Figure:
    """Heatmap showing image counts per macroclass × year."""
    df = ann[["macroclass_label", "campaign_year"]].copy()
    df["macroclass_label"] = df["macroclass_label"].apply(_safe).replace("", "Unknown")
    df["campaign_year"] = df["campaign_year"].apply(_safe_year).replace("", "Unknown")

    pivot = df.groupby(["macroclass_label", "campaign_year"]).size().unstack(fill_value=0)

    # Sort year columns numerically
    cols = list(pivot.columns)
    numeric_cols = []
    other_cols = []
    for c in cols:
        try:
            numeric_cols.append((int(c), c))
        except ValueError:
            other_cols.append(c)
    numeric_cols.sort()
    ordered = [c for _, c in numeric_cols] + sorted(other_cols)
    pivot = pivot.reindex(columns=ordered, fill_value=0)

    z_vals = pivot.values.astype(int).tolist()

    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=[str(c) for c in pivot.columns],
        y=list(pivot.index),
        colorscale="Blues",
        text=z_vals,
        texttemplate="%{text}",
        hovertemplate="Macroclass: %{y}<br>Year: %{x}<br>Count: %{z:,}<extra></extra>",
    ))
    _apply_layout(fig, "Macroclass × Year Heatmap",
                  height=max(450, 55 * len(pivot.index)))
    fig.update_xaxes(type="category", title_text="Year")
    fig.update_yaxes(type="category", title_text="")
    return fig


# ═════════════════════════════════════════════════════════════════
# Report Builder
# ═════════════════════════════════════════════════════════════════

def generate_report(output_dir: Path, save_png: bool = True) -> Path:
    """
    Generate the full statistics report.

    Parameters
    ----------
    output_dir : Path
        Root merge-output directory (contains registries/, Duplicados/, etc.)
    save_png : bool
        Also export individual PNGs into ``output_dir/stats_figures/``.

    Returns
    -------
    Path to the generated stats_report.html.
    """
    ann, dup = load_data(output_dir)
    n = len(ann)

    # Generate all figures
    figures = []

    print(f"  [1/10] Duplicates ratio …")
    figures.append(("duplicates_ratio", fig_duplicates_ratio(ann, dup)))

    print(f"  [2/10] Year distribution …")
    figures.append(("year_distribution", fig_year_distribution(ann)))

    print(f"  [3/10] Macroclass distribution …")
    figures.append(("macroclass_distribution", fig_macroclass_distribution(ann)))

    print(f"  [4/10] Taxonomy sunburst …")
    figures.append(("taxonomy_sunburst", fig_taxonomy_sunburst(ann)))

    print(f"  [5/10] Taxonomy treemap …")
    figures.append(("taxonomy_treemap", fig_taxonomy_treemap(ann)))

    print(f"  [6/10] Taxonomy Sankey …")
    figures.append(("taxonomy_sankey", fig_taxonomy_sankey(ann)))

    print(f"  [7/10] Specimen ID presence …")
    figures.append(("specimen_id_presence", fig_specimen_id_presence(ann)))

    print(f"  [8/10] Shared specimen IDs …")
    figures.append(("shared_specimen_ids", fig_shared_specimen_ids(ann)))

    print(f"  [9/10] Field completeness …")
    figures.append(("field_completeness", fig_field_completeness(ann)))

    print(f"  [10/10] Macroclass × Year heatmap …")
    figures.append(("macroclass_year_heatmap", fig_macroclass_year_heatmap(ann)))

    # ── Save PNGs ──
    if save_png:
        fig_dir = output_dir / "stats_figures"
        fig_dir.mkdir(exist_ok=True)
        for name, fig in figures:
            try:
                fig.write_image(str(fig_dir / f"{name}.png"), scale=2)
            except Exception as e:
                logger.warning(f"Could not save {name}.png: {e}")
        print(f"  PNGs saved to {fig_dir}")

    # ── Build HTML report ──
    html_parts = [_html_header(n, len(dup))]
    for name, fig in figures:
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
        html_parts.append("<hr style='border-color:#2a3040; margin: 30px 0;'>")
    html_parts.append(_html_footer())

    report_path = output_dir / "stats_report.html"
    report_path.write_text("\n".join(html_parts), encoding="utf-8")
    print(f"\n✅  Report saved to {report_path}")
    return report_path


def _html_header(n_ann: int, n_dup: int) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Dataset Statistics Report</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  body {{
    background: {BG_COLOR};
    color: {TEXT_COLOR};
    font-family: 'Segoe UI', sans-serif;
    max-width: 1200px;
    margin: 0 auto;
    padding: 30px;
  }}
  h1 {{ color: #ffffff; border-bottom: 2px solid {ACCENT}; padding-bottom: 10px; }}
  .summary {{
    display: flex; gap: 20px; flex-wrap: wrap; margin: 20px 0 30px;
  }}
  .summary .card {{
    background: {CARD_BG}; border-radius: 10px; padding: 16px 24px;
    border: 1px solid #2a3a55; min-width: 160px;
  }}
  .summary .card .num {{ font-size: 28px; font-weight: bold; color: {ACCENT}; }}
  .summary .card .label {{ font-size: 12px; color: #7a8ba5; text-transform: uppercase; }}
  hr {{ border: none; border-top: 1px solid #2a3040; }}
</style>
</head>
<body>
<h1>📊 Dataset Statistics Report</h1>
<div class="summary">
  <div class="card"><div class="num">{n_ann:,}</div><div class="label">Annotations</div></div>
  <div class="card"><div class="num">{n_dup:,}</div><div class="label">Duplicate Records</div></div>
  <div class="card"><div class="num">{n_ann + n_dup:,}</div><div class="label">Total Files</div></div>
</div>
<hr>
"""


def _html_footer() -> str:
    from datetime import datetime
    return f"""
<p style="text-align:center; color:#555; margin-top:40px; font-size:12px;">
  Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
</p>
</body></html>"""
