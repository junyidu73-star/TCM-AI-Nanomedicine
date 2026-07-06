# -*- coding: utf-8 -*-
"""
app.py —— AI 驱动的中药计算辅助药物发现平台 (CDSS)
====================================================
Streamlit 主程序：把三大后端引擎串联为一个商业级 Web 界面。

后端映射（标准 Python 包 + 绝对引入）
------------------------------------
  系统 A（精准医学）  : engine_target.discovery  +  tcm_original_books_kb.strict_mapping
  系统 B（知识发现）  : engine_target.discovery (GEO 爬虫) + tcm_original_books_kb.strict_mapping
  MD/CADD 流水线      : pipeline_md_cadd.mlmd_and_responsive + pipeline_md_cadd.docking_network

设计强调
--------
  方剂推荐**严格遵循中医经典原著/统编教材**，由 strict_mapping 的白名单知识库
  经集合运算得出，绝不使用任何生成式/幻觉式推断。
"""

from __future__ import annotations

import os
import sys

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
#  确保项目根目录在 sys.path 上，随后用最标准的绝对引入加载后端各子包。
#  三个子目录均已放置 __init__.py，成为标准 Python 包；带数字前缀的旧文件名
#  （1_/2_）已重命名为合法模块名 mlmd_and_responsive / docking_network。
# --------------------------------------------------------------------------- #
BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from engine_target import discovery
from tcm_original_books_kb import strict_mapping as strict
from pipeline_md_cadd import mlmd_and_responsive as mlmd
from pipeline_md_cadd import docking_network as docking


# --------------------------------------------------------------------------- #
#  页面配置
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="AI 驱动的中药计算辅助药物发现平台 (CDSS)",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 AI 驱动的中药计算辅助药物发现平台 (CDSS)")
st.caption("Multi-omics Target Discovery · 严格原著方剂映射 · MD / CADD 自组装流水线")

# --------------------------------------------------------------------------- #
#  侧边栏：系统架构说明
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("系统架构")
    st.markdown(
        """
**系统 A · 精准医学**
上传私有临床/组学数据 → 差异分析挖掘核心靶点 → 严格方剂映射。

**系统 B · 知识发现**
输入疾病方向 → 公共 GEO 组学抓取 → 严格方剂映射。

**MD / CADD 流水线**
自组装形态学（Core/Shell、κ²）+ 微酸响应解离 + Vina 对接 + 网络构建。
        """
    )
    st.divider()
    st.success(
        "🔒 **方剂推荐严格遵循中医经典原著**\n\n"
        "所有方剂均可溯源至《伤寒论》《医林改错》《太平惠民和剂局方》等"
        "经典典籍与全国统编教材，经**集合运算**匹配得出，"
        "**绝不使用任何大模型幻觉或自由发散**。"
    )
    st.divider()
    st.caption("环境依赖：streamlit · pandas · scikit-learn · MDAnalysis · seaborn")


def _render_matches(res: dict):
    """渲染严格方剂映射结果（含溯源）。"""
    st.info(res.get("note", ""))
    matches = res.get("matches", [])
    if matches:
        st.subheader("📜 严格匹配方剂（可溯源）")
        st.dataframe(pd.DataFrame(matches), use_container_width=True)
    downloads = res.get("downloads", [])
    if downloads:
        st.subheader("🧪 单体成分 3D 结构 (.sdf)")
        st.dataframe(pd.DataFrame(downloads), use_container_width=True)


# --------------------------------------------------------------------------- #
#  主体 Tabs
# --------------------------------------------------------------------------- #
tab1, tab2 = st.tabs(["🩺 系统 A：精准医学", "🌐 系统 B：知识发现"])

# ============================ Tab 1: 精准医学 ============================ #
with tab1:
    st.subheader("上传私有临床/组学表达数据 (CSV)")
    st.caption("格式：基因为行、样本为列，并含一行 `group` 标签（取值 case / control）。")
    up = st.file_uploader("选择 CSV 文件", type=["csv"], key="private_csv")

    df_preview = None
    if up is not None:
        try:
            df_preview = pd.read_csv(up, index_col=0)
            st.dataframe(df_preview.head(12), use_container_width=True)
        except Exception as exc:
            st.error(f"读取 CSV 失败：{exc}")

    if st.button("🚀 启动靶点挖掘与方剂推荐", type="primary", key="run_private"):
        if df_preview is None:
            st.warning("请先上传有效的 CSV 数据。")
        else:
            with st.spinner("差异表达分析中…"):
                result = discovery.analyze_private_data(df_preview)
            st.success(result.note)
            frame = result.to_frame()
            if not frame.empty:
                st.subheader("🎯 显著核心靶点")
                st.dataframe(frame, use_container_width=True)
                targets = list(frame["gene"])
                with st.spinner("严格方剂映射 + 单体结构下载中…"):
                    res = strict.get_prescription_and_download(targets, top_k=3)
                _render_matches(res)
                # 缓存靶点供流水线使用
                st.session_state["targets"] = targets
            else:
                st.info("未检出显著靶点，请调整阈值或检查数据分组。")

# ============================ Tab 2: 知识发现 ============================ #
with tab2:
    st.subheader("输入疾病方向，抓取公共组学并映射方剂")
    disease = st.text_input("疾病方向（建议英文名，便于 GEO 检索）",
                            value="colorectal cancer", key="disease")
    manual = st.text_input("（可选）已知靶点，逗号分隔，将直接用于方剂映射",
                           value="TNF, IL6, NFKB1, PTGS2", key="manual_targets")

    if st.button("🌐 启动全网公共组学抓取", type="primary", key="run_public"):
        with st.spinner(f"检索 NCBI GEO：{disease} …"):
            geo = discovery.scrape_geo_public(disease)
        st.info(geo.note)
        gf = geo.to_frame()
        if not gf.empty:
            st.subheader("📚 GEO 数据集")
            st.dataframe(gf, use_container_width=True)

        # 用手工靶点（或 GEO 候选）驱动严格方剂映射
        targets = [t.strip() for t in manual.split(",") if t.strip()]
        if not targets:
            targets = geo.candidate_targets
        if targets:
            with st.spinner("严格方剂映射 + 单体结构下载中…"):
                res = strict.get_prescription_and_download(targets, top_k=3)
            _render_matches(res)
            st.session_state["targets"] = targets
        else:
            st.warning("暂无可用靶点进行方剂映射，请在上方手工填写靶点。")

# --------------------------------------------------------------------------- #
#  全局底部：MD / CADD 流水线
# --------------------------------------------------------------------------- #
st.divider()
st.header("⚙️ MD 黑盒自组装与多靶点对接流水线")

col_a, col_b = st.columns([3, 2])
with col_a:
    gro = st.text_input("结构文件 (.gro)", value=os.path.join(BASE, "prod.gro"))
    xtc = st.text_input("轨迹文件 (.xtc)", value=os.path.join(BASE, "prod.xtc"))
with col_b:
    st.caption("流水线阶段：\n\n"
               "1. ML-MD 形态学（Core/Shell + κ²）\n"
               "2. 微酸响应电荷解离\n"
               "3. Vina 对接 + 网络构建")

if st.button("🔥 启动 MD 黑盒自组装与多靶点对接流水线",
             type="primary", use_container_width=True, key="run_pipeline"):

    # ---------- 阶段 1：ML-MD 形态学 ----------
    with st.spinner("① 读取轨迹，跑层次聚类与主惯性张量…"):
        if os.path.exists(gro) and os.path.exists(xtc):
            traj = mlmd.analyze_trajectory(gro, xtc)
            st.success("① ML-MD 形态学分析完成。")
            m1, m2, m3, m4 = st.columns(4)
            Ix, Iy, Iz = traj["principal_moments"]
            m1.metric("κ² 形状各向异性", f"{traj['kappa2']:.3f}")
            m2.metric("Iz/Ix", f"{traj['anisotropy_ratio']:.2f}")
            m3.metric("Core 组分", ", ".join(traj["core_species"]))
            m4.metric("Shell 组分", ", ".join(traj["shell_species"]))
            st.write(f"**形态判定：** {traj['verdict']}  "
                     f"（Ix:Iy:Iz = {Ix:.0f} : {Iy:.0f} : {Iz:.0f}）")
            if os.path.exists(traj["heatmap_path"]):
                st.image(traj["heatmap_path"],
                         caption="组分间 COM 距离热力图", width=520)
            core_species = traj["core_species"]
        else:
            st.warning("未找到 prod.gro / prod.xtc，跳过阶段 ①。")
            core_species = []

    # ---------- 阶段 2：微酸响应电荷解离 ----------
    with st.spinner("② 修改 .top 电荷，模拟微酸(pH≈6.5)解离…"):
        top_path = os.path.join(BASE, "topol.top")
        resp = mlmd.modify_top_for_acidic_env(top_path, target_resname="UNL",
                                              charge_delta=+1.0)
        st.success("② 微酸响应电荷修改完成。")
        st.json({k: resp[k] for k in ("atoms_modified", "net_charge_change",
                                      "output_top")})

    # ---------- 阶段 3：Vina 对接 + 网络 ----------
    with st.spinner("③ AutoDock Vina 对接与网络构建…"):
        targets = st.session_state.get("targets",
                                       ["TNF", "IL6", "NFKB1", "PTGS2", "AKT1"])
        # 从严格 KB 取单体作为配体
        kb_res = strict.get_prescription_and_download(targets, top_k=2)
        ligands = [d["monomer"] for d in kb_res.get("downloads", [])] \
            or ["Puerarin", "Baicalin", "Berberine"]
        ligand_target_map = {lg: targets for lg in ligands}
        target_pathway_map = {
            "TNF": ["TNF signaling", "NF-kB pathway"],
            "IL6": ["JAK-STAT pathway"],
            "NFKB1": ["NF-kB pathway"],
            "PTGS2": ["Arachidonic acid metabolism"],
            "AKT1": ["PI3K-AKT pathway"],
        }
        net = docking.dock_and_build_network(ligand_target_map, target_pathway_map)
        st.success(f"③ 对接完成：强结合边 {net['n_strong_bindings']} 条，"
                   f"网络总边 {net['n_edges']} 条（阈值 {net['cutoff']} kcal/mol）。")
        st.subheader("🔗 药物-靶点-通路 网络边列表 (Cytoscape)")
        st.dataframe(net["edge_list"], use_container_width=True)
        with open(net["edge_csv"], "rb") as fh:
            st.download_button("⬇️ 下载网络边列表 CSV", fh,
                               file_name="drug_target_pathway_network.csv",
                               mime="text/csv")
        st.subheader("📊 对接结合能排名")
        st.dataframe(net["dock_table"], use_container_width=True)

    st.balloons()
    st.success("✅ MD 黑盒自组装与多靶点对接流水线全部完成。")
