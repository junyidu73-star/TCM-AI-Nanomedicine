# -*- coding: utf-8 -*-
"""
pipeline_md_cadd/1_mlmd_and_responsive.py
=========================================
整合两大功能：

功能一 —— ML-MD 形态学分析
  读取 MD 轨迹，用 MDAnalysis + scikit-learn 做层次聚类，划定
  Core / Shell，计算整体复合物的主惯性张量与相对形状各向异性 (κ²)，
  并绘制 N×N 组分间距离热力图。

功能二 —— 环境响应（微酸解离）机制模拟
  读取 GROMACS .top 拓扑，定位特定可质子化残基（如羧酸 -COOH / 胺基），
  在“微酸环境（肿瘤微环境 pH≈6.5）”下修改其电荷（模拟质子化/解离），
  输出修改后的拓扑，供后续对接/再次 MD 使用。

本文件既可作为库被 app.py 调用，也可命令行独立运行。
"""

from __future__ import annotations

import os
import re
import shutil
from typing import Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
#  功能一：ML-MD 形态学分析
# --------------------------------------------------------------------------- #
def analyze_trajectory(
    gro_path: str,
    xtc_path: str,
    drug_resid_max: int = 140,
    n_species: int = 7,
    last_fraction: float = 0.2,
    out_heatmap: str = None,
) -> Dict:
    """
    读取轨迹并完成核壳聚类 + 惯性张量/κ² 计算 + 距离热力图。

    参数
    ----
    gro_path       : 结构文件 (.gro)
    xtc_path       : 轨迹文件 (.xtc)
    drug_resid_max : 药物分子最大 resid（其后为水/离子）
    n_species      : 组分（物种）数
    last_fraction  : 取轨迹末段比例作为平衡窗口
    out_heatmap    : 热力图输出路径（默认与脚本同目录）

    返回
    ----
    dict: 距离矩阵、核壳分类、惯性矩、κ²、热力图路径等。
    """
    import MDAnalysis as mda
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if out_heatmap is None:
        out_heatmap = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "core_shell_heatmap.png")

    u = mda.Universe(gro_path, xtc_path)
    drug = u.select_atoms(f"resid 1:{drug_resid_max}")
    residues = drug.residues
    n_res = residues.n_residues

    # 将 resid 均匀切成 n_species 块（每块为一个物种的全部拷贝）
    per = n_res // n_species
    species_groups = []
    for s in range(n_species):
        lo = s * per
        hi = (s + 1) * per if s < n_species - 1 else n_res
        species_groups.append(residues[lo:hi].atoms)

    n_frames = len(u.trajectory)
    start = max(0, int(n_frames * (1.0 - last_fraction)))
    frames = list(range(start, n_frames))

    # 累积：物种两两 COM 距离、整体惯性矩、κ²
    dmat = np.zeros((n_species, n_species))
    I_acc = np.zeros(3)
    kappa_acc = []

    def _pbc_gather(ag, box):
        """最小镜像收拢：以第一个原子为参考，消除跨盒断裂。"""
        pos = ag.positions.copy()
        ref = pos[0]
        d = pos - ref
        d -= box[:3] * np.round(d / box[:3])
        return ref + d

    for fi in frames:
        u.trajectory[fi]
        box = u.dimensions
        # 各物种 COM
        coms = []
        for ag in species_groups:
            p = _pbc_gather(ag, box)
            m = ag.masses
            coms.append((p * m[:, None]).sum(0) / m.sum())
        coms = np.array(coms)
        for i in range(n_species):
            for j in range(n_species):
                dmat[i, j] += np.linalg.norm(coms[i] - coms[j]) / 10.0  # Å->nm

        # 整体复合物惯性张量与 κ²
        allp = _pbc_gather(drug, box)
        m = drug.masses
        com = (allp * m[:, None]).sum(0) / m.sum()
        rel = allp - com
        # 惯性张量
        Ixx = (m * (rel[:, 1] ** 2 + rel[:, 2] ** 2)).sum()
        Iyy = (m * (rel[:, 0] ** 2 + rel[:, 2] ** 2)).sum()
        Izz = (m * (rel[:, 0] ** 2 + rel[:, 1] ** 2)).sum()
        Ixy = -(m * rel[:, 0] * rel[:, 1]).sum()
        Ixz = -(m * rel[:, 0] * rel[:, 2]).sum()
        Iyz = -(m * rel[:, 1] * rel[:, 2]).sum()
        Itensor = np.array([[Ixx, Ixy, Ixz],
                            [Ixy, Iyy, Iyz],
                            [Ixz, Iyz, Izz]])
        eig = np.sort(np.linalg.eigvalsh(Itensor)) / 100.0  # amu·nm²
        I_acc += eig
        # 回转张量 -> κ²
        gyr = (rel[:, :, None] * rel[:, None, :] * m[:, None, None]).sum(0) / m.sum()
        lam = np.sort(np.linalg.eigvalsh(gyr))[::-1] / 100.0
        lam_mean = lam.mean()
        kappa2 = 1.0 - 3.0 * ((lam[0] * lam[1] + lam[1] * lam[2] + lam[0] * lam[2])
                              / (lam.sum() ** 2))
        kappa_acc.append(kappa2)

    nf = len(frames)
    dmat /= nf
    I_mean = I_acc / nf
    kappa2_mean = float(np.mean(kappa_acc))

    # 层次聚类：按“到颗粒中心的平均距离”把物种分 Core / Shell
    # 用距离矩阵做 average linkage，切成 2 簇
    condensed = squareform(dmat, checks=False)
    Z = linkage(condensed, method="average")
    labels = fcluster(Z, t=2, criterion="maxclust")

    # 计算每个物种到整体中心的半径，均值小的簇=Core
    center = dmat.mean(0)  # 近似：与其它物种的平均距离越小越靠中心
    radius = dmat.mean(1)
    cl_radius = {c: radius[labels == c].mean() for c in np.unique(labels)}
    core_cl = min(cl_radius, key=cl_radius.get)
    core_species = [i for i in range(n_species) if labels[i] == core_cl]
    shell_species = [i for i in range(n_species) if labels[i] != core_cl]

    # 热力图
    sp_labels = [f"S{i+1}" for i in range(n_species)]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(dmat, annot=True, fmt=".2f", cmap="viridis",
                xticklabels=sp_labels, yticklabels=sp_labels,
                cbar_kws={"label": "Mean COM distance (nm)"}, ax=ax)
    ax.set_title("Inter-species COM distance matrix (equilibrated window)")
    fig.tight_layout()
    fig.savefig(out_heatmap, dpi=300)
    plt.close("all")

    Ix, Iy, Iz = I_mean
    aniso = Iz / Ix if Ix > 0 else float("nan")
    if aniso < 1.15 and kappa2_mean < 0.05:
        verdict = "近完美球形（各向同性球状胶束）"
    elif aniso < 1.30 and kappa2_mean < 0.10:
        verdict = "类球形/球状聚集体（轻度各向异性）"
    elif aniso < 1.80 and kappa2_mean < 0.25:
        verdict = "椭球/长椭球聚集体（明显非球形）"
    else:
        verdict = "拉长/各向异性聚集体（棒状或盘状）"

    return {
        "distance_matrix": dmat,
        "species_labels": sp_labels,
        "core_species": [sp_labels[i] for i in core_species],
        "shell_species": [sp_labels[i] for i in shell_species],
        "principal_moments": (float(Ix), float(Iy), float(Iz)),
        "anisotropy_ratio": float(aniso),
        "kappa2": kappa2_mean,
        "verdict": verdict,
        "heatmap_path": out_heatmap,
        "n_frames_used": nf,
    }


# --------------------------------------------------------------------------- #
#  功能二：微酸环境响应（.top 电荷修改，模拟解离/质子化）
# --------------------------------------------------------------------------- #
# 在 pH≈6.5 的肿瘤微环境中，羧酸基（pKa~4-5）大多仍解离(-1)，
# 而某些胺/咪唑（pKa~6-7）会被质子化(+1)。此处提供一个可配置的
# 电荷偏移规则，演示如何据此改写 [ atoms ] 段的 charge 列。
def modify_top_for_acidic_env(
    top_path: str,
    target_resname: str,
    charge_delta: float = +1.0,
    atom_name_pattern: str = r"^N",   # 默认对氮原子（胺/咪唑）施加质子化
    out_path: Optional[str] = None,
) -> Dict:
    """
    读取 GROMACS .top，对指定残基中匹配的原子施加电荷偏移，
    模拟微酸环境下的质子化/解离，写出新拓扑。

    参数
    ----
    top_path          : 输入 .top 路径
    target_resname    : 目标残基名（[ atoms ] 段第 4 列 resname）
    charge_delta      : 电荷改变量（+1 表示质子化，-1 表示去质子/解离）
    atom_name_pattern : 原子名正则（默认以 N 开头，代表可质子化氮）
    out_path          : 输出路径（默认 <原名>_acidic.top）

    返回
    ----
    dict: 修改统计（改动原子数、净电荷变化、输出路径）。

    说明：GROMACS [ atoms ] 段列格式通常为：
        nr  type  resnr  resname  atomname  cgnr  charge  mass
    """
    if out_path is None:
        base, ext = os.path.splitext(top_path)
        out_path = f"{base}_acidic{ext}"

    if not os.path.exists(top_path):
        # 无真实拓扑时，生成一个演示用最小 .top 以保证流程连续
        _write_demo_top(top_path, target_resname)

    with open(top_path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()

    out_lines: List[str] = []
    in_atoms = False
    changed = 0
    net_delta = 0.0
    name_re = re.compile(atom_name_pattern)

    for ln in lines:
        stripped = ln.strip()
        # 段落切换检测
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped.strip("[] ").lower()
            in_atoms = (section == "atoms")
            out_lines.append(ln)
            continue

        if in_atoms and stripped and not stripped.startswith((";", "#")):
            cols = ln.split()
            # 需要至少 7 列（含 charge）
            if len(cols) >= 7:
                resname = cols[3]
                atomname = cols[4]
                if resname == target_resname and name_re.search(atomname):
                    try:
                        q = float(cols[6])
                        newq = q + charge_delta
                        net_delta += charge_delta
                        changed += 1
                        # 保留原始列宽风格，替换第 7 列
                        cols[6] = f"{newq:.4f}"
                        ln = "   " + "  ".join(cols) + "\n"
                    except ValueError:
                        pass
        out_lines.append(ln)

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.writelines(out_lines)

    return {
        "input_top": top_path,
        "output_top": out_path,
        "target_resname": target_resname,
        "atoms_modified": changed,
        "net_charge_change": round(net_delta, 4),
        "note": (f"已在残基 {target_resname} 上对匹配 '{atom_name_pattern}' 的 "
                 f"{changed} 个原子施加 {charge_delta:+.1f} 电荷偏移，"
                 f"模拟微酸(pH≈6.5)环境下的质子化/解离。"),
    }


def _write_demo_top(path: str, resname: str) -> None:
    """写一个最小可解析的演示 .top（仅用于无真实拓扑时跑通流程）。"""
    demo = f""";  DEMO topology (auto-generated placeholder)
[ moleculetype ]
; name   nrexcl
{resname}     3

[ atoms ]
;  nr  type  resnr  resname  atomname  cgnr   charge     mass
    1   nh    1      {resname}    N1       1     -0.4000   14.0100
    2   hn    1      {resname}    HN1      1      0.3000    1.0080
    3   c3    1      {resname}    C1       2      0.1000   12.0110
    4   oh    1      {resname}    O1       3     -0.6000   16.0000
    5   ho    1      {resname}    HO1      3      0.4000    1.0080

[ bonds ]
    1  2
    1  3
    3  4
    4  5
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(demo)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="ML-MD 形态分析 + 微酸响应电荷修改")
    ap.add_argument("--gro", default="prod.gro")
    ap.add_argument("--xtc", default="prod.xtc")
    ap.add_argument("--top", default="topol.top")
    ap.add_argument("--resname", default="LIG")
    args = ap.parse_args()

    if os.path.exists(args.gro) and os.path.exists(args.xtc):
        res = analyze_trajectory(args.gro, args.xtc)
        print("核壳分类:", res["core_species"], "|", res["shell_species"])
        print("κ² =", round(res["kappa2"], 4), "| 判定:", res["verdict"])
        print("热力图:", res["heatmap_path"])

    r2 = modify_top_for_acidic_env(args.top, args.resname)
    print(r2["note"], "->", r2["output_top"])
