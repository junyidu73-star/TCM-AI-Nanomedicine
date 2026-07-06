#!/usr/bin/env bash
# =============================================================================
# pipeline_md_cadd/0_run_md_pipeline.sh
# -----------------------------------------------------------------------------
# 标准全原子 MD 制备与生产流水线模板（命令行流）
#
#   Gaussian (RESP 电荷)  ->  Antechamber (GAFF2 力场)  ->
#   GROMACS 建盒/加水/加离子  ->  能量最小化  ->  NVT 平衡  ->
#   NPT 平衡  ->  100 ns 生产 MD
#
# 说明：本脚本为“可读的标准流程模板”，其中的量化/建模步骤需要
#       Gaussian、AmberTools、GROMACS 等外部软件，请在配置好的
#       HPC/工作站环境中执行。默认 set -e，任一步失败即终止。
# =============================================================================
set -euo pipefail

# ------------------------- 0. 用户可配置参数 ---------------------------------
LIG_NAME="LIG"              # 配体残基名（3 字符）
LIG_PDB="ligand.pdb"        # 输入配体结构（已加氢、构象合理）
NET_CHARGE=0                # 配体净电荷
MULTIPLICITY=1              # 自旋多重度
NPROC=16                    # Gaussian 并行核数
MEM="32GB"                  # Gaussian 内存
BOX_TYPE="cubic"            # 盒子类型
BOX_EDGE=1.2                # 溶质距盒壁最小距离 (nm)
WATER_MODEL="tip3p"         # 水模型
FF="amber99sb-ildn"        # 蛋白/体系力场（配体用 GAFF2 叠加）
SALT_CONC=0.15             # 生理盐浓度 (mol/L)
PROD_NS=100                 # 生产 MD 时长 (ns)

echo "==> [Stage 1/6] Gaussian RESP 电荷拟合"
# -----------------------------------------------------------------------------
# 1) 生成 Gaussian 输入：HF/6-31G* 单点 + MK(RESP) 布居，输出静电势
cat > resp.gjf <<EOF
%nproc=${NPROC}
%mem=${MEM}
#p HF/6-31G* SCF=Tight Pop=MK IOp(6/33=2,6/41=10,6/42=17)

RESP charge fitting for ${LIG_NAME}

${NET_CHARGE} ${MULTIPLICITY}
$(tail -n +3 "${LIG_PDB}" 2>/dev/null | awk '/^(ATOM|HETATM)/{printf "%-2s %12.6f %12.6f %12.6f\n",$3,$6,$7,$8}')

EOF
# 运行 Gaussian（生成 resp.log / resp.chk）
g16 resp.gjf
# 由 Gaussian 输出提取 RESP 电荷（AmberTools 的 antechamber/respgen 亦可）
antechamber -i resp.log -fi gout -o ${LIG_NAME}.mol2 -fo mol2 \
            -c resp -nc ${NET_CHARGE} -rn ${LIG_NAME} -at gaff2

echo "==> [Stage 2/6] Antechamber / GAFF2 力场参数化"
# -----------------------------------------------------------------------------
# 2) 生成缺失力场参数 + 转成 GROMACS 可用拓扑（借助 acpype）
parmchk2 -i ${LIG_NAME}.mol2 -f mol2 -o ${LIG_NAME}.frcmod -s gaff2
# 生成 Amber prmtop/inpcrd
cat > tleap.in <<EOF
source leaprc.gaff2
LIG = loadmol2 ${LIG_NAME}.mol2
loadamberparams ${LIG_NAME}.frcmod
saveamberparm LIG ${LIG_NAME}.prmtop ${LIG_NAME}.inpcrd
quit
EOF
tleap -f tleap.in
# Amber -> GROMACS （acpype 或 ParmEd）
acpype -p ${LIG_NAME}.prmtop -x ${LIG_NAME}.inpcrd -b ${LIG_NAME}

echo "==> [Stage 3/6] GROMACS 建盒 / 溶剂化 / 加离子"
# -----------------------------------------------------------------------------
gmx editconf -f ${LIG_NAME}_GMX.gro -o box.gro -bt ${BOX_TYPE} -d ${BOX_EDGE} -c
gmx solvate  -cp box.gro -cs spc216.gro -p topol.top -o solv.gro
gmx grompp   -f ions.mdp -c solv.gro -p topol.top -o ions.tpr -maxwarn 2
echo "SOL" | gmx genion -s ions.tpr -o solv_ions.gro -p topol.top \
                        -pname NA -nname CL -neutral -conc ${SALT_CONC}

echo "==> [Stage 4/6] 能量最小化 (EM)"
# -----------------------------------------------------------------------------
gmx grompp -f em.mdp -c solv_ions.gro -p topol.top -o em.tpr -maxwarn 2
gmx mdrun  -v -deffnm em

echo "==> [Stage 5/6] NVT 平衡 (100 ps) 与 NPT 平衡 (100 ps)"
# -----------------------------------------------------------------------------
# NVT：位置限制 + V-rescale 控温至 310 K
gmx grompp -f nvt.mdp -c em.gro -r em.gro -p topol.top -o nvt.tpr -maxwarn 2
gmx mdrun  -v -deffnm nvt
# NPT：Parrinello-Rahman 控压至 1 bar
gmx grompp -f npt.mdp -c nvt.gro -r nvt.gro -t nvt.cpt -p topol.top -o npt.tpr -maxwarn 2
gmx mdrun  -v -deffnm npt

echo "==> [Stage 6/6] 生产 MD (${PROD_NS} ns)"
# -----------------------------------------------------------------------------
# md.mdp 中 nsteps = ${PROD_NS} ns / 2 fs = $((PROD_NS * 500000)) 步
gmx grompp -f md.mdp -c npt.gro -t npt.cpt -p topol.top -o prod.tpr -maxwarn 2
gmx mdrun  -v -deffnm prod

# 生产后 PBC 处理（unwrap + 居中），供下游 ML-MD 分析
echo "System" | gmx trjconv -s prod.tpr -f prod.xtc -o prod_whole.xtc -pbc whole
echo "==> 流水线完成：prod.xtc / prod.gro 可用于 mlmd_and_responsive.py"
