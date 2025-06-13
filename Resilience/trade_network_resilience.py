"""
trade_network_resilience.py  –  fast eff_w edition
==================================================

• 仅计算加权全局效率 (eff_w)
• 预计算 Floyd–Warshall 距离矩阵，大幅降低删点重算开销
• 生成:  ① 节点冲击热图 ② 年度鲁棒性曲线 ③ AUC 趋势折线
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


# ───────────────────────────────────────────────────────────────────────────
# CONFIG                                                                     
# ───────────────────────────────────────────────────────────────────────────
DATA_SOURCE   = "E:\Obsidian\Project\专业课\大三下\Intro to Networks\Final_Essay\IMF_IMTS_Exports_1948_2024.csv"  # or dir of CSVs
COORD_CSV     = "E:\Obsidian\Project\专业课\大三下\Intro to Networks\Final_Essay\iso3_to_latlon.csv"
YEAR_START    = 2015
YEAR_END      = 2024
IMPACT_TOP_N  = 30
THRESHOLD     = None
OUTPUT_DIR    = Path("outputs"); OUTPUT_DIR.mkdir(exist_ok=True)

# ──── user-tunable display settings ────────────────────────────────────────
HEATMAP_MODE  = "rank"       # "quantile" | "row_norm" | "rank"
GAMMA         = 0.4
Q_LOW, Q_HIGH = .05, .95
TOP_K_NODES   = 20
COLORMAP      = "Reds"


# ───────────────────────────────────────────────────────────────────────────
# 0. Coordinates                                                             
# ───────────────────────────────────────────────────────────────────────────
def _load_coord_dict(fp: str | Path) -> Dict[str, tuple[float, float]]:
    df = pd.read_csv(fp)
    return {r.ISO3: (r.Lat, r.Lon)
            for _, r in df.iterrows()
            if pd.notna(r.Lat) and pd.notna(r.Lon)}

COORD      = _load_coord_dict(COORD_CSV)
ISO_VALID  = set(COORD.keys())

# ───────────────────────────────────────────────────────────────────────────
# 1. Graph loader                                                            
# ───────────────────────────────────────────────────────────────────────────
def load_trade_graph(year: int,
                     src: Path | str,
                     threshold: float | None = None) -> nx.Graph:
    """Return weighted undirected trade graph for one year."""
    src = Path(src)
    if src.is_dir():
        df = (pd.read_csv(src / f"exports_{year}.csv")
                .rename(columns={"exporter": "u",
                                 "importer": "v",
                                 "value":    "w"}))
    else:
        raw = pd.read_csv(src, low_memory=False)
        if str(year) not in raw.columns:
            raise ValueError(f"{year} not in {src.name}")
        df = raw.loc[:, ["COUNTRY.ID", "COUNTERPART_COUNTRY.ID", str(year)]]
        df.columns = ["u", "v", "w"]

    df = (df.dropna(subset=["w"])
            .query("w>0 and u!=v")
            .loc[lambda x: x.u.isin(ISO_VALID) & x.v.isin(ISO_VALID)])
    if threshold is not None:
        df = df.query("w >= @threshold")

    G = nx.Graph()
    for u, v, w in df.itertuples(index=False):
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    return G
# ─── 2. 距离矩阵 & 效率计算 ────────────────────────────────────────────────
def distance_matrix(G: nx.Graph) -> tuple[np.ndarray, list]:
    """返回 Floyd-Warshall 最短距离矩阵 (1/weight) 及节点顺序列表。"""
    W = nx.to_numpy_array(G, weight="weight", dtype=float)
    W[W == 0] = np.inf
    D = 1 / W
    np.fill_diagonal(D, 0)
    dist = nx.floyd_warshall_numpy(nx.from_numpy_array(D))
    return dist, list(G)

def efficiency_from_mask(dist: np.ndarray, mask: np.ndarray) -> float:
    """给定距阵 dist 和保留节点布尔 mask，返回加权全局效率。"""
    sub = dist[np.ix_(mask, mask)]
    with np.errstate(divide='ignore'):
        inv = 1.0 / sub
    np.fill_diagonal(inv, 0)
    n = sub.shape[0]
    return np.sum(inv) / (n*(n-1)) if n > 1 else 0.0

# ── 新增指标函数 ─────────────────────────────────────────
def weighted_eigenvector(G: nx.Graph) -> Dict[str, float]:
    return nx.eigenvector_centrality_numpy(G, weight="weight")

def weighted_pagerank(G: nx.Graph) -> Dict[str, float]:
    return nx.pagerank(G, weight="weight", alpha=0.85)

def weighted_betweenness(G: nx.Graph) -> Dict[str, float]:
    return nx.betweenness_centrality(G, weight=lambda u,v,d: 1/d["weight"],
                                     normalized=True)

def weighted_kcore(G: nx.Graph) -> Dict[str, int]:
    # strength-core: Opsahl et al. 2010 —— 把加权度代入 k-core 删除规则
    H = G.copy()
    core = {n: 0 for n in H}
    k = 1
    while H:
        to_remove = [n for n,d in H.degree(weight='weight') if d < k]
        if not to_remove:
            k += 1; continue
        for n in to_remove:
            core[n] = k-1
            H.remove_node(n)
    # 返回归一化到 [0,1]
    max_k = max(core.values())
    return {n: core[n]/max_k for n in core}

CENT_FUNS = {
    "eig": weighted_eigenvector,
    "pr" : weighted_pagerank,
    "bc" : weighted_betweenness,
    "kcore": weighted_kcore,
}

# ── 通用绘图：年度 top-K 热图 ────────────────────────────
def yearly_centrality_heatmap(metric_key: str, years: range,
                              top_k: int = 20, cmap=COLORMAP):
    rows = []
    for y in years:
        G = load_trade_graph(y, DATA_SOURCE)
        cent = CENT_FUNS[metric_key](G)
        top = sorted(cent, key=cent.get, reverse=True)[:top_k]
        rows += [{"year": y, "country": c, "score": cent[c]} for c in top]

    df = pd.DataFrame(rows)
    pivot = (df.pivot(index="year", columns="country", values="score")
               .fillna(0)
               .sort_index())

    plt.figure(figsize=(14,6))
    sns.heatmap(pivot, cmap=cmap, norm=mpl.colors.Normalize(0, pivot.values.max()),
                linewidths=.3, linecolor="0.9",
                cbar_kws={"label": f"{metric_key} centrality"})
    plt.title(f"Weighted {metric_key.upper()} centrality heat-map")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"heatmap_{metric_key}.png", dpi=300)
    plt.close()
    print("[✓] heatmap saved:", f"heatmap_{metric_key}.png")

# ─── 3. 单节点冲击 (矩阵版) ────────────────────────────────────────────────
def single_node_impact_fast(G: nx.Graph,
                            top_n: int = IMPACT_TOP_N) -> Dict[str, float]:
    dist, nodes = distance_matrix(G)
    n_total     = len(nodes)
    mask_all    = np.ones(n_total, dtype=bool)
    base_eff    = efficiency_from_mask(dist, mask_all)

    # 前 top_n 按加权度排序
    deg = {n: G.degree(n, weight="weight") for n in nodes}
    top_nodes = sorted(nodes, key=deg.get, reverse=True)[:top_n]

    impacts = {}
    for node in top_nodes:
        idx        = nodes.index(node)
        mask       = mask_all.copy();  mask[idx] = False
        new_eff    = efficiency_from_mask(dist, mask)
        impacts[node] = (base_eff - new_eff) / base_eff
    return impacts

# ─── 4. 鲁棒性曲线 (矩阵版) ────────────────────────────────────────────────
def robustness_curve_fast(G: nx.Graph) -> pd.DataFrame:
    dist, nodes = distance_matrix(G)
    strength    = {n: G.degree(n, weight="weight") for n in nodes}
    nodes_sorted = sorted(nodes, key=strength.get, reverse=True)

    mask = np.ones(len(nodes), dtype=bool)
    base = efficiency_from_mask(dist, mask)
    curve = []
    for i, node in enumerate(nodes_sorted, 1):
        mask[nodes.index(node)] = False
        val = efficiency_from_mask(dist, mask) if mask.sum() > 1 else 0.0
        curve.append({"removed_frac": i/len(nodes), "metric": val/base})
    return pd.DataFrame(curve)

auc = lambda df: np.trapz(df.metric, df.removed_frac)

# ─── 5. 可视化 ────────────────────────────────────────────────────────────
def plot_heatmap(df_imp: pd.DataFrame):
    mode = HEATMAP_MODE.lower()
    pivot = (df_imp
             .groupby("year").apply(lambda g: g.nlargest(TOP_K_NODES, "impact"))
             .droplevel(0)
             .pivot(index="year", columns="country", values="impact")
             .fillna(0))

    if mode == "rank":                               # ── rank 逻辑 ──
        rank = pivot.rank(axis=1, method="min", ascending=False)
        data = (rank - 1) / (TOP_K_NODES - 1)        # 0(最好)-1(最差)
        cmap = COLORMAP
        norm = mpl.colors.Normalize(0, 1)
        cbar_ticks  = np.linspace(0, 1, 5)
        cbar_labels = [f"{int(t*(TOP_K_NODES-1))+1}" for t in cbar_ticks]
        cbar_label  = f"Rank (1 = 最高冲击)"

    else:                                            # ── quantile 逻辑 ──
        data = pivot.copy()
        vmin, vmax = np.quantile(data.values, Q_LOW), np.quantile(data.values, Q_HIGH)
        cmap = COLORMAP
        norm = mpl.colors.PowerNorm(gamma=GAMMA, vmin=vmin, vmax=vmax)
        cbar_ticks  = np.linspace(vmin, vmax, 5)
        cbar_labels = [f"{x:.2f}" for x in cbar_ticks]
        cbar_label  = f"Impact value (γ = {GAMMA})"

    plt.figure(figsize=(13, 6))
    ax = sns.heatmap(data, cmap=cmap, norm=norm,
                     linecolor="0.9", linewidths=.4,
                     cbar_kws={"label": cbar_label})

    # 重新写颜色条刻度
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels(cbar_labels)

    plt.title(f"Weighted efficiency impact heat-map – {mode}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"heatmap_eff_w_{mode}.png", dpi=300)
    plt.close()
    print(f"[✓] heatmap saved: heatmap_eff_w_{mode}.png")

def plot_curve(df: pd.DataFrame, year: int):
    plt.figure()
    plt.plot(df.removed_frac, df.metric, lw=2)
    plt.ylim(0,1.02); plt.grid(ls="--", alpha=.4)
    plt.title(f"Robustness curve eff_w – {year}")
    plt.xlabel("fraction removed"); plt.ylabel("metric/base")
    plt.tight_layout()
    plt.close()

def plot_auc_trend(df_auc: pd.DataFrame):
    plt.figure(figsize=(6,4))
    sns.lineplot(df_auc, x="year", y="auc", marker="o")
    plt.ylabel("AUC (↑ = more robust)")
    plt.title("Yearly robustness (AUC) – eff_w")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"auc_trend_eff_w.png", dpi=300)
    plt.close()
    print("[✓] trend plot saved")

# ─── 6. 主流程 ────────────────────────────────────────────────────────────
def run_pipeline():
    years = range(YEAR_START, YEAR_END+1)

    # ① 单节点冲击热图
    impact_rows = []
    for y in years:
        G = load_trade_graph(y, DATA_SOURCE)
        imp = single_node_impact_fast(G, IMPACT_TOP_N)
        impact_rows += [{"year": y, "country": c, "impact": v} for c, v in imp.items()]
    df_imp = pd.DataFrame(impact_rows)
    df_imp.to_csv(OUTPUT_DIR/"node_impacts_eff_w_fast.csv", index=False)
    plot_heatmap(df_imp)

    # # ② 鲁棒性 & AUC 趋势
    # auc_rows = []
    # for y in years:
    #     G     = load_trade_graph(y, DATA_SOURCE)
    #     curve = robustness_curve_fast(G)
    #     plot_curve(curve, y)
    #     auc_rows.append({"year": y, "auc": auc(curve)})
    # df_auc = pd.DataFrame(auc_rows)
    # df_auc.to_csv(OUTPUT_DIR/"auc_eff_w_fast.csv", index=False)
    # plot_auc_trend(df_auc)


# ── 主入口调用示例 ───────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["impact", "centrality", "all"],
                        default="impact")
    args = parser.parse_args()

    yrs = range(YEAR_START, YEAR_END+1)

    if args.task in ("impact", "all"):
        run_pipeline()

    if args.task in ("centrality", "all"):
        for key in ["eig", "pr", "bc"]:
            yearly_centrality_heatmap(key, yrs, top_k=TOP_K_NODES)
