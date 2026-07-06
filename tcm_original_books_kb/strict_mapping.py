# -*- coding: utf-8 -*-
"""
tcm_original_books_kb/strict_mapping.py
=======================================
系统核心中的核心：严格方剂-靶点映射引擎。

最高设计指令（Hard Constraint）
--------------------------------
本模块在把“疾病/靶点”映射到“中药方剂”时，**绝不允许**任何大模型自由发散、
臆测或“幻觉式”推荐。所有推荐必须可溯源到一条**结构化知识库记录**，
而该记录直接引用中医经典原著、经典名方与全国统编教材的原始出处
（书名 + 篇目/条文）。

实现方式
--------
1) 知识库以**白名单规则表**（KNOWLEDGE_BASE）硬编码/加载于本文件，
   每条规则包含： 方剂名、出处典籍、原文条文、主治、组成、
   以及该方“现代研究已知作用靶点/通路”的**受控词表**。
2) 映射函数只做**集合运算与检索**（靶点集合 ∩ 规则靶点集合），
   不调用任何生成式模型，从算法层面杜绝幻觉。
3) 任何无法在知识库中溯源的匹配，一律拒绝输出，并返回明确的“无匹配”。

数据结构
--------
Prescription（方剂）
  name           : 方剂名（如 “葛根芩连汤”）
  source_book    : 出处典籍（如 “《伤寒论》”）
  source_passage : 原文/条文出处（如 “辨太阳病脉证并治”）
  indication     : 主治（原著表述）
  composition    : 组成（君臣佐使药材列表）
  known_targets  : 现代研究支持的作用靶点（受控词表，用于与组学靶点求交集）
  monomers       : 代表性单体成分（含 PubChem CID，用于下载 3D 结构）
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
#  数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Monomer:
    """方剂中的代表性单体化合物。"""
    name: str                    # 中文/英文名
    pubchem_cid: Optional[int]   # PubChem CID（用于 API 下载 3D SDF）
    role: str = ""               # 归属药材/作用简述


@dataclass
class Prescription:
    """一条严格可溯源的方剂知识库记录。"""
    name: str
    source_book: str
    source_passage: str
    indication: str
    composition: List[str]
    known_targets: List[str]
    monomers: List[Monomer] = field(default_factory=list)

    def citation(self) -> str:
        return f"{self.name}（{self.source_book}·{self.source_passage}）"


@dataclass
class MappingHit:
    """一次严格匹配的结果。"""
    prescription: Prescription
    matched_targets: List[str]         # 命中的交集靶点
    score: float                       # 匹配得分（交集靶点数 / 归一化）
    evidence: str                      # 溯源说明（出处 + 命中靶点）


# --------------------------------------------------------------------------- #
#  严格知识库（白名单）——所有条目均引用经典原著/统编教材
#  说明：known_targets 为“现代药理研究已明确报道”的受控词表，
#        仅用于与组学靶点求交集，不代表原著本身使用现代靶点术语。
# --------------------------------------------------------------------------- #
KNOWLEDGE_BASE: List[Prescription] = [
    Prescription(
        name="葛根芩连汤",
        source_book="《伤寒论》",
        source_passage="辨太阳病脉证并治·第34条",
        indication="太阳病，桂枝证，医反下之，利遂不止，脉促，喘而汗出者",
        composition=["葛根", "黄芩", "黄连", "炙甘草"],
        known_targets=["TNF", "IL6", "NFKB1", "PTGS2", "TLR4", "IL1B"],
        monomers=[
            Monomer("葛根素 (Puerarin)", 5281807, "葛根，君药"),
            Monomer("黄芩苷 (Baicalin)", 64982, "黄芩，臣药"),
            Monomer("小檗碱 (Berberine)", 2353, "黄连，臣药"),
        ],
    ),
    Prescription(
        name="黄连解毒汤",
        source_book="《肘后备急方》（《外台秘要》引崔氏方）",
        source_passage="崔氏方·疗伤寒时气温病",
        indication="一切实热火毒，三焦热盛之证",
        composition=["黄连", "黄芩", "黄柏", "栀子"],
        known_targets=["NFKB1", "PTGS2", "TNF", "IL6", "NOS2", "HMOX1"],
        monomers=[
            Monomer("小檗碱 (Berberine)", 2353, "黄连，君药"),
            Monomer("黄芩苷 (Baicalin)", 64982, "黄芩，臣药"),
            Monomer("栀子苷 (Geniposide)", 107848, "栀子，佐药"),
        ],
    ),
    Prescription(
        name="血府逐瘀汤",
        source_book="《医林改错》",
        source_passage="卷上·血府逐瘀汤所治之症目",
        indication="胸中血瘀，血行不畅，胸痛头痛日久，痛如针刺而有定处",
        composition=["桃仁", "红花", "当归", "生地黄", "川芎", "赤芍",
                     "牛膝", "桔梗", "柴胡", "枳壳", "甘草"],
        known_targets=["VEGFA", "PTGS2", "MMP9", "TNF", "IL6", "PPARG", "HIF1A"],
        monomers=[
            Monomer("羟基红花黄色素A (Hydroxysafflor yellow A)", 6443665, "红花"),
            Monomer("阿魏酸 (Ferulic acid)", 445858, "当归/川芎"),
            Monomer("芍药苷 (Paeoniflorin)", 442534, "赤芍"),
        ],
    ),
    Prescription(
        name="六味地黄丸",
        source_book="《小儿药证直诀》",
        source_passage="卷下·地黄丸",
        indication="肾怯失音，囟开不合，神不足，目中白睛多，面色㿠白等肾阴虚证",
        composition=["熟地黄", "山茱萸", "山药", "泽泻", "牡丹皮", "茯苓"],
        known_targets=["AR", "IGF1", "TGFB1", "AKT1", "PPARG", "NR3C1"],
        monomers=[
            Monomer("莫诺苷 (Morroniside)", 11228693, "山茱萸"),
            Monomer("丹皮酚 (Paeonol)", 11092, "牡丹皮"),
            Monomer("泽泻醇B (Alisol B)", 21599926, "泽泻"),
        ],
    ),
    Prescription(
        name="补阳还五汤",
        source_book="《医林改错》",
        source_passage="卷下·瘫痿论·补阳还五汤",
        indication="中风之气虚血瘀证，半身不遂，口眼㖞斜，语言謇涩",
        composition=["黄芪", "当归尾", "赤芍", "地龙", "川芎", "红花", "桃仁"],
        known_targets=["VEGFA", "HIF1A", "AKT1", "BDNF", "TNF", "CASP3", "NOS3"],
        monomers=[
            Monomer("黄芪甲苷 (Astragaloside IV)", 13943297, "黄芪，君药"),
            Monomer("阿魏酸 (Ferulic acid)", 445858, "当归尾/川芎"),
            Monomer("芍药苷 (Paeoniflorin)", 442534, "赤芍"),
        ],
    ),
    Prescription(
        name="麻黄汤",
        source_book="《伤寒论》",
        source_passage="辨太阳病脉证并治·第35条",
        indication="外感风寒表实证，恶寒发热，无汗而喘，脉浮紧",
        composition=["麻黄", "桂枝", "杏仁", "炙甘草"],
        known_targets=["ADRB2", "PTGS2", "TNF", "IL6", "NFKB1"],
        monomers=[
            Monomer("麻黄碱 (Ephedrine)", 9294, "麻黄，君药"),
            Monomer("苦杏仁苷 (Amygdalin)", 656516, "杏仁，佐药"),
            Monomer("桂皮醛 (Cinnamaldehyde)", 637511, "桂枝，臣药"),
        ],
    ),
    Prescription(
        name="逍遥散",
        source_book="《太平惠民和剂局方》",
        source_passage="卷之九·治妇人诸疾·逍遥散",
        indication="肝郁血虚脾弱证，两胁作痛，头痛目眩，神疲食少，月经不调",
        composition=["柴胡", "当归", "白芍", "白术", "茯苓", "炙甘草", "生姜", "薄荷"],
        known_targets=["HTR1A", "SLC6A4", "BDNF", "TNF", "IL6", "NR3C1", "MAOA"],
        monomers=[
            Monomer("柴胡皂苷a (Saikosaponin A)", 167928, "柴胡，君药"),
            Monomer("芍药苷 (Paeoniflorin)", 442534, "白芍，臣药"),
            Monomer("阿魏酸 (Ferulic acid)", 445858, "当归"),
        ],
    ),
]


# --------------------------------------------------------------------------- #
#  严格映射：只做集合运算，绝不生成
# --------------------------------------------------------------------------- #
def _normalize_target(t: str) -> str:
    """靶点名归一化：去空格、转大写，便于受控词表精确匹配。"""
    return str(t).strip().upper()


def strict_match(
    target_list: List[str],
    kb: Optional[List[Prescription]] = None,
    min_overlap: int = 1,
) -> List[MappingHit]:
    """
    将组学靶点列表严格映射到知识库方剂。

    算法（无任何生成式推断）：
      对每条方剂记录 P：
        overlap = set(靶点) ∩ set(P.known_targets)
        若 |overlap| >= min_overlap，则命中；
        得分 score = |overlap| / |P.known_targets|（覆盖率，[0,1]）。
      所有命中按 (交集大小 desc, score desc) 排序返回。

    任何不在 KNOWLEDGE_BASE 受控词表中的“想当然”匹配都不会产生，
    因为交集运算天然拒绝词表外的靶点。
    """
    kb = kb if kb is not None else KNOWLEDGE_BASE
    query = {_normalize_target(t) for t in (target_list or []) if str(t).strip()}
    if not query:
        return []

    hits: List[MappingHit] = []
    for pres in kb:
        kb_targets = {_normalize_target(t) for t in pres.known_targets}
        overlap = sorted(query & kb_targets)
        if len(overlap) >= min_overlap:
            score = len(overlap) / max(1, len(kb_targets))
            evidence = (
                f"依据 {pres.citation()}；"
                f"原著主治：{pres.indication}；"
                f"命中受控靶点：{', '.join(overlap)}。"
            )
            hits.append(MappingHit(
                prescription=pres,
                matched_targets=overlap,
                score=round(score, 4),
                evidence=evidence,
            ))

    hits.sort(key=lambda h: (len(h.matched_targets), h.score), reverse=True)
    return hits


# --------------------------------------------------------------------------- #
#  PubChem 单体 3D 结构下载（真实 API + 降级模拟）
# --------------------------------------------------------------------------- #
def _download_sdf_from_pubchem(
    cid: int,
    out_path: str,
    timeout: int = 20,
) -> bool:
    """
    调用 PubChem PUG REST 下载指定 CID 的 3D 构象 SDF。

    真实接口：
      https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/<CID>/SDF?record_type=3d

    成功写入返回 True；失败返回 False（由上层决定是否降级为占位）。
    """
    try:
        import requests
    except Exception:
        return False

    url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
           f"{cid}/SDF?record_type=3d")
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200 and r.text.strip().endswith("$$$$"):
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(r.text)
            return True
        return False
    except Exception:
        return False


def _write_placeholder_sdf(monomer: Monomer, out_path: str) -> None:
    """
    无网络时写一个占位 SDF（明确标注为占位，不含真实坐标），
    保证下游流水线的文件路径连续性，同时避免把假坐标当真数据。
    """
    content = (
        f"{monomer.name}\n"
        f"  PLACEHOLDER  PubChem CID={monomer.pubchem_cid}\n"
        f"  [占位文件] 未能联网下载真实 3D 结构，请在生产环境重试。\n"
        f"  0  0  0  0  0  0  0  0  0  0999 V2000\n"
        f"M  END\n"
        f"$$$$\n"
    )
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(content)


def get_prescription_and_download(
    target_list: List[str],
    download_dir: str = None,
    top_k: int = 3,
    min_overlap: int = 1,
) -> Dict:
    """
    主入口：接收组学靶点 -> 严格匹配方剂 -> 下载其单体 3D 结构 (.sdf)。

    参数
    ----
    target_list  : 组学分析得到的靶点列表（基因符号）
    download_dir : SDF 下载目录（默认 本模块目录/downloaded_sdf）
    top_k        : 取匹配度最高的前 K 个方剂做结构下载
    min_overlap  : 判定命中所需的最小靶点交集数

    返回
    ----
    dict:
      {
        "matches": [ {方剂溯源信息...}, ... ],
        "downloads": [ {name, cid, path, status}, ... ],
        "note": 说明
      }

    严格性保证：matches 全部来自 strict_match（集合运算），
    不存在任何生成式/幻觉式方剂。
    """
    if download_dir is None:
        download_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "downloaded_sdf")
    os.makedirs(download_dir, exist_ok=True)

    hits = strict_match(target_list, min_overlap=min_overlap)
    if not hits:
        return {
            "matches": [],
            "downloads": [],
            "note": ("在严格知识库中未找到与所给靶点可溯源匹配的方剂。"
                     "按最高指令，系统拒绝生成任何无出处的推荐。"),
        }

    selected = hits[:top_k]
    matches_out = []
    downloads_out = []

    for h in selected:
        pres = h.prescription
        matches_out.append({
            "prescription": pres.name,
            "source": f"{pres.source_book}·{pres.source_passage}",
            "indication": pres.indication,
            "composition": "、".join(pres.composition),
            "matched_targets": ", ".join(h.matched_targets),
            "score": h.score,
            "evidence": h.evidence,
        })

        for mono in pres.monomers:
            safe_name = "".join(c for c in mono.name if c.isalnum() or c in " _-()")
            fname = f"{pres.name}_{safe_name}_CID{mono.pubchem_cid}.sdf"
            fpath = os.path.join(download_dir, fname)
            ok = False
            if mono.pubchem_cid:
                ok = _download_sdf_from_pubchem(mono.pubchem_cid, fpath)
            if not ok:
                _write_placeholder_sdf(mono, fpath)
            downloads_out.append({
                "prescription": pres.name,
                "monomer": mono.name,
                "cid": mono.pubchem_cid,
                "path": fpath,
                "status": "downloaded" if ok else "placeholder",
            })
            time.sleep(0.2)  # 尊重 PubChem 频率限制

    n_ok = sum(1 for d in downloads_out if d["status"] == "downloaded")
    return {
        "matches": matches_out,
        "downloads": downloads_out,
        "note": (f"严格匹配到 {len(hits)} 个方剂，选取前 {len(selected)} 个下载单体结构；"
                 f"真实下载 {n_ok}/{len(downloads_out)} 个 SDF"
                 f"（其余为占位文件，需联网重试）。"),
    }


if __name__ == "__main__":
    # 自测：给一组炎症相关靶点
    demo_targets = ["TNF", "IL6", "NFKB1", "PTGS2"]
    res = get_prescription_and_download(demo_targets, top_k=2)
    print(res["note"])
    for m in res["matches"]:
        print("-", m["prescription"], "|", m["source"], "|", m["matched_targets"])
