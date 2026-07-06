# -*- coding: utf-8 -*-
"""
engine_target/discovery.py
==========================
系统 A（精准医学）后端核心：多组学靶点发现引擎。

本模块提供两条互补的靶点发现路径：
  1) analyze_private_data(df)   —— 基于用户上传的“私有”临床/组学表达矩阵，
                                    做差异表达分析，挖掘显著失调的核心靶点。
  2) scrape_geo_public(disease) —— 基于疾病名，从公共 GEO 数据库检索相关数据集，
                                    给出可复用的公共靶点检索逻辑骨架。

设计原则
--------
* 纯函数、可单测：analyze_private_data 只依赖传入的 DataFrame，不读写全局状态。
* 统计稳健：差异分析使用独立样本 t 检验 + Benjamini-Hochberg (BH) FDR 校正，
  避免仅凭 fold-change 产生大量假阳性。
* 失败安全：网络爬虫在无网络/接口变更时返回结构化的降级结果，而非抛栈崩溃，
  以便上层 Streamlit UI 稳定展示。
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
#  数据结构定义
# --------------------------------------------------------------------------- #
@dataclass
class TargetHit:
    """单个差异靶点的结构化结果。"""
    gene: str                 # 基因/靶点名
    log2fc: float             # log2 fold change（病例组 vs 对照组）
    p_value: float            # 原始 p 值
    fdr: float                # BH 校正后的 q 值 (FDR)
    mean_case: float          # 病例组平均表达
    mean_ctrl: float          # 对照组平均表达
    direction: str            # "UP" 上调 / "DOWN" 下调

    def as_dict(self) -> dict:
        return {
            "gene": self.gene,
            "log2FC": round(self.log2fc, 4),
            "p_value": self.p_value,
            "FDR": self.fdr,
            "mean_case": round(self.mean_case, 4),
            "mean_ctrl": round(self.mean_ctrl, 4),
            "direction": self.direction,
        }


@dataclass
class DiscoveryResult:
    """analyze_private_data 的整体返回。"""
    hits: List[TargetHit] = field(default_factory=list)
    n_genes_tested: int = 0
    n_significant: int = 0
    note: str = ""

    def to_frame(self) -> pd.DataFrame:
        if not self.hits:
            return pd.DataFrame(
                columns=["gene", "log2FC", "p_value", "FDR",
                         "mean_case", "mean_ctrl", "direction"]
            )
        return pd.DataFrame([h.as_dict() for h in self.hits])


# --------------------------------------------------------------------------- #
#  1. 私有数据差异分析
# --------------------------------------------------------------------------- #
def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR 校正。

    参数
    ----
    pvals : 原始 p 值数组

    返回
    ----
    q 值（FDR）数组，与输入等长、次序一致。
    """
    p = np.asarray(pvals, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)                 # 升序索引
    ranked = p[order]
    # BH 公式： q_i = p_i * n / rank_i ，再做单调化处理
    q = ranked * n / (np.arange(1, n + 1))
    # 从后往前取累计最小，保证 q 值单调不减
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty_like(q)
    out[order] = q
    return out


def analyze_private_data(
    df: pd.DataFrame,
    group_row_label: str = "group",
    case_label: str = "case",
    ctrl_label: str = "control",
    fdr_threshold: float = 0.05,
    log2fc_threshold: float = 1.0,
    top_n: Optional[int] = 50,
) -> DiscoveryResult:
    """
    对私有表达矩阵做差异表达分析，返回显著的核心靶点。

    期望输入格式（两种兼容布局，自动识别）
    -----------------------------------------
    布局 A（推荐，基因为行、样本为列）：
        第一列为基因名（index 或名为 gene/Gene/symbol 的列）；
        其余列为样本；额外存在一行标签行 group_row_label（如 "group"），
        其取值为 case_label / ctrl_label 标注每个样本所属分组。

    布局 B（样本为行、基因为列）：
        存在名为 group_row_label 的列标注分组，其余数值列为基因表达。
        本函数会自动转置为布局 A 处理。

    统计方法
    --------
    * 每个基因在 case / control 两组间做 Welch t 检验（不假设方差齐性）。
    * log2FC = log2(mean_case + eps) - log2(mean_ctrl + eps)。
    * 多重检验用 BH-FDR 校正。
    * 显著性判定： FDR < fdr_threshold 且 |log2FC| >= log2fc_threshold。

    返回
    ----
    DiscoveryResult
    """
    from scipy import stats  # 延迟导入，避免无 scipy 时整模块不可用

    if df is None or df.empty:
        return DiscoveryResult(note="输入数据为空。")

    work = df.copy()

    # --- 统一为“基因为行、样本为列” + 一行 group 标签 ---
    # 情况 B：group 是一列 -> 转置
    lower_cols = {str(c).strip().lower(): c for c in work.columns}
    if group_row_label.lower() in lower_cols and group_row_label.lower() != "group_is_index":
        gcol = lower_cols[group_row_label.lower()]
        # 若 group 在列中且样本在行 -> 需要转置为标准布局
        # 判定：数值列占多数说明是“样本为行、基因为列”
        numeric_like = work.drop(columns=[gcol]).apply(
            lambda s: pd.to_numeric(s, errors="coerce").notna().mean()
        )
        if (numeric_like > 0.8).mean() > 0.5:
            groups = work[gcol].astype(str).str.strip().str.lower()
            expr = work.drop(columns=[gcol]).apply(pd.to_numeric, errors="coerce")
            expr = expr.T  # 基因为行、样本为列
            group_series = groups
            group_series.index = df.index  # 样本名
        else:
            expr, group_series = _parse_layout_a(work, group_row_label)
    else:
        expr, group_series = _parse_layout_a(work, group_row_label)

    if expr is None or group_series is None:
        return DiscoveryResult(
            note=(f"未能识别分组信息。请确保存在标签行/列 '{group_row_label}'，"
                  f"取值包含 '{case_label}' 与 '{ctrl_label}'。")
        )

    g = group_series.astype(str).str.strip().str.lower()
    case_cols = g[g == case_label.lower()].index
    ctrl_cols = g[g == ctrl_label.lower()].index

    if len(case_cols) < 2 or len(ctrl_cols) < 2:
        return DiscoveryResult(
            note=(f"每组至少需要 2 个样本才能做 t 检验（当前 case={len(case_cols)}, "
                  f"control={len(ctrl_cols)}）。")
        )

    expr_num = expr.apply(pd.to_numeric, errors="coerce")
    case_mat = expr_num[case_cols].values.astype(float)
    ctrl_mat = expr_num[ctrl_cols].values.astype(float)

    eps = 1e-9
    mean_case = np.nanmean(case_mat, axis=1)
    mean_ctrl = np.nanmean(ctrl_mat, axis=1)
    log2fc = np.log2(np.abs(mean_case) + eps) - np.log2(np.abs(mean_ctrl) + eps)

    # 向量化 Welch t 检验
    with np.errstate(all="ignore"):
        tstat, pvals = stats.ttest_ind(
            case_mat, ctrl_mat, axis=1, equal_var=False, nan_policy="omit"
        )
    pvals = np.asarray(pvals, dtype=float)
    pvals[np.isnan(pvals)] = 1.0
    fdr = _bh_fdr(pvals)

    genes = list(expr_num.index)
    hits: List[TargetHit] = []
    for i, gene in enumerate(genes):
        sig = (fdr[i] < fdr_threshold) and (abs(log2fc[i]) >= log2fc_threshold)
        if not sig:
            continue
        hits.append(TargetHit(
            gene=str(gene),
            log2fc=float(log2fc[i]),
            p_value=float(pvals[i]),
            fdr=float(fdr[i]),
            mean_case=float(mean_case[i]),
            mean_ctrl=float(mean_ctrl[i]),
            direction="UP" if log2fc[i] > 0 else "DOWN",
        ))

    # 按 FDR 升序、|log2FC| 降序排序
    hits.sort(key=lambda h: (h.fdr, -abs(h.log2fc)))
    n_sig = len(hits)
    if top_n is not None:
        hits = hits[:top_n]

    return DiscoveryResult(
        hits=hits,
        n_genes_tested=len(genes),
        n_significant=n_sig,
        note=(f"共检验 {len(genes)} 个基因，显著靶点 {n_sig} 个"
              f"（FDR<{fdr_threshold}, |log2FC|>={log2fc_threshold}）。"),
    )


def _parse_layout_a(work: pd.DataFrame, group_row_label: str):
    """
    解析“基因为行、样本为列，且含一行 group 标签行”的布局。
    返回 (expr_df[基因x样本], group_series[样本->分组])。
    """
    # 找到基因名列
    gene_col = None
    for cand in ("gene", "Gene", "symbol", "Symbol", "GENE"):
        if cand in work.columns:
            gene_col = cand
            break
    if gene_col is not None:
        work = work.set_index(gene_col)

    # 找 group 标签行
    idx_lower = {str(i).strip().lower(): i for i in work.index}
    if group_row_label.lower() not in idx_lower:
        return None, None
    grow = idx_lower[group_row_label.lower()]
    group_series = work.loc[grow]
    expr = work.drop(index=grow)
    return expr, group_series


# --------------------------------------------------------------------------- #
#  2. 公共 GEO 数据检索（爬虫骨架）
# --------------------------------------------------------------------------- #
@dataclass
class GeoResult:
    disease: str
    datasets: List[dict] = field(default_factory=list)   # [{accession, title, n_samples, url}]
    candidate_targets: List[str] = field(default_factory=list)
    source: str = ""
    note: str = ""

    def to_frame(self) -> pd.DataFrame:
        if not self.datasets:
            return pd.DataFrame(columns=["accession", "title", "n_samples", "url"])
        return pd.DataFrame(self.datasets)


def scrape_geo_public(
    disease_name: str,
    retmax: int = 10,
    timeout: int = 15,
) -> GeoResult:
    """
    基于疾病名从 NCBI GEO (Gene Expression Omnibus) 检索相关数据集。

    实现路径
    --------
    使用 NCBI E-utilities 公共接口（无需登录）：
      1) esearch：在 gds（GEO DataSets）库中按疾病名检索，取回 GEO 记录 UID 列表。
      2) esummary：批量取回每个 UID 的标题、样本数、GSE 号等元信息。

    这是一个“可运行的爬虫骨架”：
      * 若本地有网络且接口可达，返回真实检索到的数据集列表；
      * 若无网络/接口变更/超时，则捕获异常并返回结构化降级结果，
        附带一份该疾病的常见候选靶点占位（明确标注为占位，需人工核实）。

    合规提示
    --------
    NCBI 要求批量访问时提供 email / api_key 并控制频率（<=3 req/s）。
    生产环境请在 params 中补充 email 与 api_key，并加入限速与重试。
    """
    disease_name = (disease_name or "").strip()
    if not disease_name:
        return GeoResult(disease="", note="疾病名为空。")

    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    try:
        import requests  # 延迟导入
    except Exception:
        return _geo_fallback(disease_name, reason="未安装 requests 库")

    try:
        # --- Step 1: esearch ---
        esearch = f"{base}/esearch.fcgi"
        params = {
            "db": "gds",
            "term": f'{disease_name}[Title] AND "Homo sapiens"[Organism]',
            "retmax": str(retmax),
            "retmode": "json",
            # 生产环境请填写真实 email / api_key：
            # "email": "you@lab.org", "api_key": "xxxx",
        }
        r = requests.get(esearch, params=params, timeout=timeout)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return GeoResult(
                disease=disease_name,
                source="NCBI GEO (esearch)",
                note="接口可达，但未检索到匹配数据集。请尝试更换疾病英文名或同义词。",
            )

        # --- Step 2: esummary ---
        esummary = f"{base}/esummary.fcgi"
        r2 = requests.get(
            esummary,
            params={"db": "gds", "id": ",".join(ids), "retmode": "json"},
            timeout=timeout,
        )
        r2.raise_for_status()
        res = r2.json().get("result", {})
        datasets = []
        for uid in res.get("uids", []):
            item = res.get(uid, {})
            acc = item.get("accession", "")
            datasets.append({
                "accession": acc,
                "title": item.get("title", ""),
                "n_samples": item.get("n_samples", ""),
                "url": f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={acc}" if acc else "",
            })

        return GeoResult(
            disease=disease_name,
            datasets=datasets,
            candidate_targets=[],   # 真实靶点应从数据集下载表达矩阵后经差异分析获得
            source="NCBI GEO E-utilities",
            note=(f"检索到 {len(datasets)} 个 GEO 数据集。下一步：下载 supplementary "
                  f"表达矩阵，交给 analyze_private_data() 做差异分析以获得靶点。"),
        )

    except Exception as exc:  # 网络/解析失败 -> 降级
        return _geo_fallback(disease_name, reason=str(exc))


def _geo_fallback(disease_name: str, reason: str) -> GeoResult:
    """无网络或接口异常时的结构化降级返回（占位靶点需人工核实）。"""
    return GeoResult(
        disease=disease_name,
        datasets=[],
        candidate_targets=[],
        source="fallback",
        note=(f"[降级模式] 未能实时访问 NCBI GEO（原因：{reason}）。"
              f"请检查网络或在生产环境配置 email/api_key。"
              f"此模式不提供占位靶点，以避免误导性数据进入下游分析。"),
    )


if __name__ == "__main__":
    # 简单自测：构造一个基因x样本 + group 行的小矩阵
    rng = np.random.default_rng(0)
    genes = [f"GENE{i}" for i in range(20)]
    cols = [f"S{i}" for i in range(10)]
    data = rng.normal(5, 1, size=(20, 10))
    data[0, :5] += 4   # GENE0 在 case 组显著上调
    data[1, :5] -= 4   # GENE1 在 case 组显著下调
    df = pd.DataFrame(data, index=genes, columns=cols)
    grow = pd.DataFrame(
        [["case"] * 5 + ["control"] * 5], index=["group"], columns=cols
    )
    df = pd.concat([df, grow])
    out = analyze_private_data(df)
    print(out.note)
    print(out.to_frame().head())
