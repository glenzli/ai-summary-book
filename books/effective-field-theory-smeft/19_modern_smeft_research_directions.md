# 第十九章：现代 SMEFT 研究方向地图

## 本章目标

本章把现代 SMEFT 的主要研究方向整理成可审查的教材地图。它不替代专门综述，而给出每一方向进入严格 EFT 分析时必须说明的算符族、尺度、截断和外部输入。

## 依赖前置知识

需要第十三章的 Warsaw basis、第十四章的 flavor/CP 纪律、第十五章的 RGE、第十六章的破缺相输入方案和第十七章的报告标准。

## 19.1 方向分类表

| 方向 | 典型观测量 | 主要算符族 | 严格性检查 |
| --- | --- | --- | --- |
| 电弱精密与 $Z$-pole | $m_W$、$Z$ partial widths、asymmetries | ${\cal O}_{HD}$、${\cal O}_{HWB}$、${\cal O}_{Hf}$ | 输入方案、LEP/SLC 协方差、universal 与 non-universal 假设 |
| Higgs 物理 | 产生截面、衰变宽度、信号强度 | ${\cal O}_{HG}$、${\cal O}_{HW}$、${\cal O}_{HB}$、${\cal O}_{HWB}$、Yukawa 型算符 | 产生和衰变共同拟合、SM 高阶修正、线性截断 |
| Top 物理 | $t\bar t$、single top、top decay | ${\cal O}_{uG}$、${\cal O}_{Hq}^{(3)}$、四夸克算符 | flavor 指标、PDF 与高能 bin 有效性 |
| 高能散射 | dilepton、diboson、VBS、高 $p_T$ tails | 四费米子、规范-Higgs current、维数八算符 | $E/\Lambda$ 截断、unitarity、维数六平方项 |
| Quark flavor | rare decays、mixing、charged currents | semileptonic 四费米子、dipole、current 算符 | SMEFT 到 LEFT 匹配、CKM 约定、hadronic matrix elements |
| Lepton flavor | LFV decay、universality tests | lepton current、semileptonic、dipole | flavor off-diagonal 分量、实验上限统计口径 |
| EDM 与 CP violation | electron/neutron/atomic EDM | CP-odd bosonic、fermion dipole、four-fermion | CP 相、RGE mixing、低能矩阵元 |
| $(g-2)_{\mu,e}$ | anomalous magnetic moments | lepton dipole、semileptonic tensor | chirality flip、matching scale、QED/QCD running |
| Neutrino 与 LNV | neutrino mass、$0\nu\beta\beta$ | Weinberg 维数五、LNV 高维算符 | Majorana 相、低能核矩阵元、维数阶数 |
| 全局拟合 | 多通道 Wilson 约束 | 上述全部的子集 | 基、尺度、flavor、协方差、工具版本和有效性切割 |

## 19.2 成熟结果与前沿结果

**定义 19.1（成熟 SMEFT 结果）.** 一个 SMEFT 结果称为成熟，若它至少给出：

1.  EFT 类型和算符基；
2.  Wilson 系数定义尺度；
3.  flavor 与 CP 假设；
4.  输入参数方案；
5.  截断阶数；
6.  RGE 与匹配处理；
7.  数据协方差或误差模型；
8.  有效性切割或能区说明。

**定义 19.2（前沿 SMEFT 结果）.** 若结果研究的是新观测量、新维数阶、新拟合策略或新 UV 匹配，但定义 19.1 中至少一项仍未闭合，则本书称其为前沿结果。前沿结果可作为研究方向，不作为本教材的基础定理输入。

## 19.3 如何阅读一个现代 SMEFT 结果

给定一篇声称限制 Wilson 系数的结果，应按以下顺序拆解：

1.  它限制的是 SMEFT、HEFT、LEFT 还是某个 simplified model；
2.  Wilson 系数在哪个尺度和哪个基中给出；
3.  数据是低能、Z-pole、Higgs、top 还是高能 tail；
4.  flavor 是否 full、diagonal、universal 或 MFV；
5.  是否只开一个系数；
6.  是否运行和匹配；
7.  是否给出协方差；
8.  是否检查 EFT validity。

**例 19.3（高质量 dilepton tail）.** 若分析使用 $pp\to\ell^+\ell^-$ 高质量 bins 并限制 semileptonic 四费米子算符，则最低元数据包括 partonic energy proxy、最大 dilepton invariant mass、PDF 设置、是否保留 $C^2/\Lambda^4$、flavor 组合和 quark chirality。缺少任一项，结果就不能直接与 flavor 或 LEP 约束合并。

## 19.4 SMEFT、HEFT、LEFT 的边界

**原则 19.4.** 高尺度新物理若尊重线性电弱实现，且低能自由度只含标准模型场，则优先用 SMEFT。若 Higgs 不作为严格 $SU(2)_L$ 双重态的线性成员组织，则应考虑 HEFT。若能量低于电弱尺度并积掉 $W,Z,h,t$，则应使用 LEFT。

**警告 19.5.** 同一篇现象学论文可能同时使用 SMEFT 和 LEFT。严格写法必须给出二者的匹配尺度，不能把 LEFT Wilson 系数直接称为 SMEFT Wilson 系数。

## 19.5 现代方向的书内收口

本书对现代方向采取三层纳入：

1.  **核心层。** EFT 定义、匹配、幂计数、RGE、EOM、Warsaw basis、flavor/CP 和破缺相输入方案。
2.  **应用层。** Higgs、电弱、top、高能散射、flavor、EDM、$g-2$ 和 LNV 的算符地图。
3.  **研究边界层。** 维数八全基、完整全局拟合、NLO SMEFT 自动化、非线性 HEFT 与低能 hadronic matrix elements。

**结论 19.6.** 现代 SMEFT 已不是单一算符表，而是跨尺度、多观测量和多统计假设的计算框架。教材必须把“算符定义”与“分析元数据”放在同等严格的位置。

## 本章小结

本章覆盖了当前 SMEFT 的主要研究方向，并给出每一方向进入严格教材主线时的最低元数据要求。这样可以避免把综述性材料误写成已闭合定理。

## 练习

**练习 19.1.** 选取 Higgs signal strength 中一个通道，列出它至少依赖哪些 SMEFT 元数据。

**练习 19.2.** 解释为什么高 $p_T$ dilepton 约束需要显式报告有效性切割。

**练习 19.3.** 给出一个同时需要 SMEFT 和 LEFT 的 flavor 观测量，并标出匹配尺度。

**练习 19.4.** 对一个高质量 dilepton SMEFT 结果列出八项元数据检查。
