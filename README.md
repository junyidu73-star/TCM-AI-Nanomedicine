<div align="center">

# 🧬 AI 驱动的中药多组学计算辅助药物发现平台 (CDSS)

**Multi-omics Target Discovery · 严格原著方剂映射 · MD / CADD 自组装流水线**

<p>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Streamlit" src="https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
  <img alt="MDAnalysis" src="https://img.shields.io/badge/MD-MDAnalysis-6E4C9F">
  <img alt="CADD" src="https://img.shields.io/badge/CADD-AutoDock%20Vina-1f77b4">
  <img alt="Zero Hallucination" src="https://img.shields.io/badge/方剂推荐-零幻觉·可溯源-2ea043">
</p>

*从「组学靶点」到「经典名方」，每一步都可溯源、可复现、无大模型幻觉。*

</div>

---

## 📖 项目简介

**CDSS**（Computational Drug-discovery Support System）是一个把**多组学靶点发现**、**严格中医经典方剂映射**与**分子动力学 / 计算机辅助药物设计（MD / CADD）流水线**串联成一体的开源研究平台。

平台采用 **双引擎架构（Dual-Engine Architecture）**：

| 引擎 | 名称 | 输入 | 核心逻辑 |
| :--- | :--- | :--- | :--- |
| **系统 A** | 🩺 精准医学引擎 | 用户上传的私有临床 / 组学表达矩阵 (CSV) | Welch t 检验 + BH-FDR 校正 → 显著核心靶点 → 严格方剂映射 |
| **系统 B** | 🌐 知识发现引擎 | 疾病方向 / 已知靶点 | NCBI GEO 公共组学检索 → 靶点 → 严格方剂映射 |

两大引擎在得到靶点后，共同汇入平台最核心的 **严格方剂映射模块** 与下游的 **MD / CADD 流水线**，形成从「病」到「靶」到「方」再到「分子机制」的完整计算闭环。

---

## ✨ 核心功能特性

### 🔒 方剂推荐严格遵循中医经典原著，绝无大模型幻觉

这是本平台**最高设计指令（Hard Constraint）**，也是与市面上「用大模型直接生成方剂」类工具的根本区别：

- **白名单知识库**：所有方剂均以结构化记录硬编码于 `tcm_original_books_kb/strict_mapping.py`，每条记录都标注**出处典籍 + 原文条文 + 主治 + 组成 + 受控靶点词表**。
- **可溯源**：推荐的每一个方剂都能溯源至《伤寒论》《医林改错》《太平惠民和剂局方》《小儿药证直诀》等经典典籍与全国统编教材的原始出处。
- **纯集合运算**：靶点到方剂的映射**只做集合的交集运算**（组学靶点 ∩ 方剂受控靶点词表），**不调用任何生成式模型**，从算法层面根除幻觉。
- **拒绝即拒绝**：任何无法在知识库中溯源的匹配，一律返回明确的「无匹配」，绝不臆造。

> 例：靶点集合 `{TNF, IL6, NFKB1, PTGS2}` 命中《伤寒论·辨太阳病脉证并治·第34条》**葛根芩连汤**，命中受控靶点即为溯源依据。

### 🩺 系统 A · 精准医学引擎

- 上传私有表达矩阵（基因为行、样本为列，含 `group` 标签行，取值 `case` / `control`）。
- 自动识别数据布局并转置，逐基因做 **Welch t 检验**（不假设方差齐性）。
- **Benjamini-Hochberg FDR** 多重检验校正，避免仅凭 fold-change 产生假阳性。
- 输出按 `FDR` 升序、`|log2FC|` 降序排序的显著核心靶点。

### 🌐 系统 B · 知识发现引擎

- 基于疾病英文名，通过 **NCBI E-utilities**（`esearch` + `esummary`）检索公共 GEO 数据集。
- **失败安全**：无网络 / 接口变更时返回结构化降级结果，而非崩溃；且不注入占位靶点，避免误导下游分析。

### ⚙️ MD 黑盒自组装与多靶点对接流水线

1. **ML-MD 形态学分析**：读取 MD 轨迹，层次聚类划定 **Core / Shell**，计算主惯性张量与相对形状各向异性 **κ²**，判定纳米粒球形度，并绘制组分间 COM 距离热力图。严格的 PBC 处理（Bai-Breen 圆周均值投影 + 最小镜像收拢）保证跨周期边界分子的完整性。
2. **微酸响应电荷解离**：模拟肿瘤微环境（pH≈6.5）下可质子化残基的电荷偏移，改写 GROMACS `.top` 拓扑。
3. **Vina 对接 + 网络构建**：对严格 KB 释放的中药单体成分做 AutoDock Vina 对接，筛选强结合（< -7.0 kcal/mol），输出 Cytoscape 可直接导入的 **药物-靶点-通路** 三层网络边列表。

> 说明：Vina 对接与 PubChem / GEO 网络请求均带有**确定性模拟回退**或**降级模式**，保证在无外部二进制 / 无网络的环境下流程仍可完整跑通并产出规范文件。

---

## 🗂️ 项目结构

```
aimd-ml-md/
├── app.py                          # Streamlit 主程序（双引擎 + 流水线 UI）
├── requirements.txt                # 部署依赖
├── engine_target/                  # 靶点发现引擎
│   └── discovery.py                #   私有数据差异分析 + GEO 公共检索
├── tcm_original_books_kb/          # ★ 严格方剂映射（白名单知识库，零幻觉）
│   └── strict_mapping.py
├── pipeline_md_cadd/               # MD / CADD 流水线
│   ├── mlmd_and_responsive.py      #   形态学分析 + 微酸响应电荷修改
│   └── docking_network.py          #   Vina 对接 + 网络构建
├── topol.top / topol_acidic.top    # 演示用 GROMACS 拓扑
└── data_uploads/ · results/        # 运行时数据（默认被 .gitignore 忽略）
```

---

## 🚀 本地运行指南

### 1. 环境要求

- Python **3.10** 或 **3.11**
- 建议使用独立虚拟环境（`venv` 或 `conda`）

### 2. 获取代码

```bash
git clone <your-repo-url>
cd aimd-ml-md
```

### 3. 创建虚拟环境并安装依赖

**使用 venv：**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

**或使用 conda：**

```bash
conda create -n cdss python=3.11 -y
conda activate cdss
pip install -r requirements.txt
```

### 4. 启动应用

```bash
streamlit run app.py
```

浏览器将自动打开 `http://localhost:8501`。

### 5. 快速体验

- **系统 A**：上传一个 CSV（基因为行、样本为列，含 `group` 标签行），点击「启动靶点挖掘与方剂推荐」。
- **系统 B**：输入疾病英文名（如 `colorectal cancer`）或直接填入已知靶点，点击「启动全网公共组学抓取」。
- **流水线**：填入 `.gro` / `.xtc` 路径（缺失时对应阶段会自动跳过或使用演示拓扑），点击「启动 MD 黑盒自组装与多靶点对接流水线」。

---

## ☁️ 云端部署（Streamlit Community Cloud）

1. 将本仓库推送到 GitHub。
2. 登录 [share.streamlit.io](https://share.streamlit.io)，选择本仓库与 `app.py` 作为入口。
3. Streamlit Cloud 会自动读取 `requirements.txt` 安装依赖并构建应用。

> 提示：如需在云端存放 NCBI email / api_key 等敏感配置，请使用 Streamlit 的 **Secrets** 管理（对应 `.streamlit/secrets.toml`，已在 `.gitignore` 中排除，切勿提交到仓库）。

---

## 🔭 未来展望

CDSS 当前完成了「从组学到方剂再到分子机制」的**计算闭环**。下一阶段，我们希望把这条计算链路延伸进**真实的物理世界**：

- **🤖 结合真实机械臂**：将平台推荐的单体组合与配比，直接下发给自动化移液 / 配液机械臂，实现「计算推荐 → 自动配制」的无人化衔接。
- **🧫 对接湿实验室（Wet-Lab）**：把 MD/CADD 预测的强结合组合送入高通量筛选与细胞 / 分子实验，用真实实验数据反向校准知识库的受控靶点词表与对接打分。
- **🔁 干湿闭环（Dry-Wet Loop）**：让湿实验结果持续回流，形成「计算预测 ↔ 实验验证」的自我迭代闭环，逐步把这套严格、可溯源的中药新药发现范式推向可落地的产业化。

---

## ⚠️ 免责声明

本平台为**科研与计算辅助**工具，所有输出（靶点、方剂、对接结果）仅供研究参考，**不构成任何医疗、诊断或用药建议**。方剂的临床使用须遵循执业中医师的辨证论治与现行法规。

---

<div align="center">

*Built for rigorous, traceable, hallucination-free TCM drug discovery.*

</div>
