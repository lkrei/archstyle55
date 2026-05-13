from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def top_k_bar(items: list[dict], title: str = "Top-5 prediction") -> go.Figure:
    df = pd.DataFrame(items)
    df = df.sort_values("prob", ascending=True)
    fig = px.bar(df, x="prob", y="cls", orientation="h",
                 text=df["prob"].map(lambda v: f"{v:.3f}"))
    fig.update_layout(title=title, height=320, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_range=[0, 1], yaxis_title=None, xaxis_title="probability")
    fig.update_traces(textposition="outside")
    return fig


def umap_atlas(points: np.ndarray, labels: list[str], hover: list[str] | None = None,
               title: str = "DINOv2 → UMAP atlas") -> go.Figure:
    df = pd.DataFrame({
        "x": points[:, 0],
        "y": points[:, 1],
        "label": labels,
        "hover": hover or labels,
    })
    fig = px.scatter(
        df, x="x", y="y", color="label", hover_name="hover",
        height=720,
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7, line=dict(width=0)))
    fig.update_layout(
        title=title,
        legend=dict(orientation="h", y=-0.15, font=dict(size=10)),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def palette_strip(items: list[dict]) -> go.Figure:
    fig = go.Figure()
    cum = 0.0
    for item in items:
        share = item["share"]
        fig.add_trace(go.Bar(
            x=[share], y=["palette"], orientation="h",
            marker_color=item["hex"], name=item["hex"],
            text=f"{item['hex']} · {share:.2f}", textposition="inside",
            insidetextanchor="middle", showlegend=False,
        ))
        cum += share
    fig.update_layout(barmode="stack", height=120, margin=dict(l=10, r=10, t=20, b=10),
                      xaxis_range=[0, 1], xaxis_title=None, yaxis_title=None,
                      yaxis=dict(showticklabels=False))
    return fig


def confidence_table(df: pd.DataFrame) -> go.Figure:
    return go.Figure(data=[go.Table(
        header=dict(values=list(df.columns), fill_color="#3070b0", font=dict(color="white"),
                    align="left"),
        cells=dict(values=[df[c] for c in df.columns], align="left",
                   fill_color="#f5f7fa"),
    )])
