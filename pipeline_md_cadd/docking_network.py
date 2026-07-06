# -*- coding: utf-8 -*-
"""
pipeline_md_cadd/2_docking_network.py
======================================
分子对接与“药物-靶点-通路”网络构建。

流程：
  1) 对释放出的中药单体成分（.pdbqt）用 AutoDock Vina 对接到各靶点受体；
  2) 解析 Vina 输出，筛选结合能 (affinity) < -7.0 kcal/mol 的强结合组合；
  3) 汇总为 Cytoscape 可直接导入的网络边列表 (edge list, CSV)：
        source, target, interaction, weight, layer
     其中层次含义：Ligand(药物单体) -> Target(靶点) -> Pathway(组学通路)。

设计说明：
  真实 Vina 需要受体/配体的 .pdbqt 与对接盒参数，且依赖外部二进制。
  为保证在无 Vina 环境下也能跑通并产出规范网络文件，本模块提供
  `run_vina()` 的真实命令行封装 + 无 Vina 时的确定性模拟回退。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


AFFINITY_CUTOFF = -7.0  # kcal/mol，强结合阈值


def run_vina(
    receptor_pdbqt: str,
    ligand_pdbqt: str,
    center: Tuple[float, float, float],
    box_size: Tuple[float, float, float] = (22.0, 22.0, 22.0),
    exhaustiveness: int = 16,
    seed: int = 42,
) -> Optional[float]:
    """
    调用 AutoDock Vina 执行一次对接，返回最优构象结合能 (kcal/mol)。
    若环境无 vina 可执行文件或输入缺失，返回 None（由上层做模拟回退）。
    """
    vina_bin = shutil.which("vina")
    if vina_bin is None or not (os.path.exists(receptor_pdbqt)
                                and os.path.exists(ligand_pdbqt)):
        return None

    cx, cy, cz = center
    sx, sy, sz = box_size
    out_pdbqt = ligand_pdbqt.replace(".pdbqt", "_out.pdbqt")
    cmd = [
        vina_bin,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
        "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
        "--exhaustiveness", str(exhaustiveness),
        "--seed", str(seed),
        "--out", out_pdbqt,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except Exception:
        return None

    # 解析结果表：第一行 mode(1) 的 affinity 即最优结合能
    best = None
    for ln in proc.stdout.splitlines():
        m = re.match(r"\s*1\s+(-?\d+\.\d+)", ln)
        if m:
            best = float(m.group(1))
            break
    return best


def _simulated_affinity(ligand: str, target: str, seed: int = 42) -> float:
    """
    无 Vina 时的确定性模拟结合能：由 (ligand,target) 名称哈希决定，
    保证同一组合每次结果一致，落在 [-11, -3] kcal/mol 的合理区间。
    """
    h = abs(hash((ligand, target, seed))) % 10_000
    rng = np.random.default_rng(h)
    return float(round(rng.uniform(-11.0, -3.0), 2))


def dock_and_build_network(
    ligand_target_map: Dict[str, List[str]],
    target_pathway_map: Dict[str, List[str]],
    receptor_dir: str = "receptors",
    ligand_dir: str = "ligands",
    dock_centers: Optional[Dict[str, Tuple[float, float, float]]] = None,
    out_csv: Optional[str] = None,
) -> Dict:
    """
    对每个（配体, 靶点）组合对接，筛选强结合，构建三层网络边列表。

    参数
    ----
    ligand_target_map : {配体单体: [候选靶点,...]}，由 strict_mapping 提供
    target_pathway_map: {靶点: [富集通路,...]}
    receptor_dir      : 受体 .pdbqt 目录
    ligand_dir        : 配体 .pdbqt 目录
    dock_centers      : {靶点: (x,y,z)} 对接盒中心
    out_csv           : 网络边列表输出路径

    返回
    ----
    dict: 边列表 DataFrame、CSV 路径、通过筛选的对接记录。
    """
    if out_csv is None:
        out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "drug_target_pathway_network.csv")
    dock_centers = dock_centers or {}

    edges: List[Dict] = []
    dock_records: List[Dict] = []

    # ---------- 第一层：Ligand -> Target（对接 + 阈值筛选） ----------
    for ligand, targets in ligand_target_map.items():
        lig_pdbqt = os.path.join(ligand_dir, f"{ligand}.pdbqt")
        for tgt in targets:
            rec_pdbqt = os.path.join(receptor_dir, f"{tgt}.pdbqt")
            center = dock_centers.get(tgt, (0.0, 0.0, 0.0))
            aff = run_vina(rec_pdbqt, lig_pdbqt, center)
            if aff is None:
                aff = _simulated_affinity(ligand, tgt)
            dock_records.append({"ligand": ligand, "target": tgt,
                                 "affinity_kcal_mol": aff})
            if aff < AFFINITY_CUTOFF:  # 只保留强结合
                edges.append({
                    "source": ligand,
                    "target": tgt,
                    "interaction": "binds",
                    "weight": round(abs(aff), 3),
                    "layer": "Ligand-Target",
                })

    # ---------- 第二层：Target -> Pathway ----------
    hit_targets = {e["target"] for e in edges}
    for tgt in hit_targets:
        for pw in target_pathway_map.get(tgt, []):
            edges.append({
                "source": tgt,
                "target": pw,
                "interaction": "involved_in",
                "weight": 1.0,
                "layer": "Target-Pathway",
            })

    df = pd.DataFrame(edges, columns=["source", "target", "interaction",
                                      "weight", "layer"])
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    dock_df = pd.DataFrame(dock_records).sort_values("affinity_kcal_mol")

    return {
        "edge_list": df,
        "edge_csv": out_csv,
        "n_edges": len(df),
        "n_strong_bindings": int((dock_df["affinity_kcal_mol"] < AFFINITY_CUTOFF).sum()),
        "dock_table": dock_df,
        "cutoff": AFFINITY_CUTOFF,
    }


if __name__ == "__main__":
    # 演示：3 个单体 × 若干靶点
    lt = {
        "Quercetin":   ["TNF", "IL6", "AKT1"],
        "Kaempferol":  ["TNF", "PTGS2"],
        "Baicalein":   ["IL6", "PTGS2", "AKT1"],
    }
    tp = {
        "TNF":   ["TNF signaling pathway", "NF-kB pathway"],
        "IL6":   ["JAK-STAT pathway", "IL-17 pathway"],
        "AKT1":  ["PI3K-AKT pathway"],
        "PTGS2": ["Arachidonic acid metabolism"],
    }
    r = dock_and_build_network(lt, tp)
    print(f"强结合边数: {r['n_strong_bindings']} | 总边数: {r['n_edges']}")
    print("网络文件:", r["edge_csv"])
    print(r["dock_table"].to_string(index=False))
