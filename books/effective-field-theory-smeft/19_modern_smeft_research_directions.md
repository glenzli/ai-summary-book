# 第十九章：SMEFT 的跨尺度研究版图

同一组 SMEFT 系数可以被 $Z$-pole、Higgs、top、高能散射和低能 flavor 数据同时探测，但这些通道并不共享同一理论接口。电弱精密量首先依赖输入参数反解，高能尾部必须控制局域与插入展开，$b\to s\ell\ell$ 还要在电弱阈值匹配到 LEFT 并引入强子矩阵元。把这些方向简单列成热点会遮蔽它们真正的计算差异。这里以“观测尺度如何连接到 Warsaw 系数”为主轴，比较每类问题的主要算符族、匹配链、flavor/CP 假设和截断风险。一个 high-mass dilepton 案例贯穿阅读方法：从外态手征性和 PDF 权重，到 $Q/M_{\rm gap}$ 切割与协方差，逐项判断结果能否与其他数据放进同一个 Wilson 空间。

## 19.1 方向分类表

| 方向 | 典型观测量 | 主要算符族 | 必要理论接口 |
| --- | --- | --- | --- |
| 电弱精密与 $Z$-pole | $m_W$、$Z$ partial widths、asymmetries | ${\cal O}_{HD}$、${\cal O}_{HWB}$、${\cal O}_{Hf}$ | 输入方案、LEP/SLC 协方差、universal 与 non-universal 假设 |
| Higgs 物理 | 产生截面、衰变宽度、信号强度 | ${\cal O}_{HG}$、${\cal O}_{HW}$、${\cal O}_{HB}$、${\cal O}_{HWB}$、Yukawa 型算符 | 产生和衰变共同拟合、SM 高阶修正、线性截断 |
| Top 物理 | $t\bar t$、single top、top decay | ${\cal O}_{uG}$、${\cal O}_{Hq}^{(3)}$、四夸克算符 | flavor 指标、PDF 与高能 bin 有效性 |
| 高能散射 | dilepton、diboson、VBS、高 $p_T$ tails | 四费米子、规范-Higgs current、维数八算符 | $Q/M_{\rm gap}$、Wilson 插入层级、unitarity、维数六平方项 |
| Quark flavor | rare decays、mixing、charged currents | semileptonic 四费米子、dipole、current 算符 | SMEFT 到 LEFT 匹配、CKM 约定、hadronic matrix elements |
| Lepton flavor | LFV decay、universality tests | lepton current、semileptonic、dipole | flavor off-diagonal 分量、实验上限统计口径 |
| EDM 与 CP violation | electron/neutron/atomic EDM | CP-odd bosonic、fermion dipole、four-fermion | CP 相、RGE mixing、低能矩阵元 |
| $(g-2)_{\mu,e}$ | anomalous magnetic moments | lepton dipole、semileptonic tensor | chirality flip、matching scale、QED/QCD running |
| Neutrino 与 LNV | neutrino mass、$0\nu\beta\beta$ | Weinberg 维数五、LNV 高维算符 | Majorana 相、低能核矩阵元、维数阶数 |
| 全局拟合 | 多通道 Wilson 约束 | 上述全部的子集 | 基、尺度、flavor、协方差、工具版本和有效性切割 |

## 19.2 哪些结果可以跨通道组合

**定义 19.1（可组合的 SMEFT 结果）.** 若一项约束要与另一能区或另一过程的结果
放入同一 Wilson 空间，至少需要共同指定：

1.  EFT 类型和算符基；
2.  Wilson 系数定义尺度；
3.  flavor 与 CP 假设；
4.  输入参数方案；
5.  截断阶数；
6.  RGE 与匹配处理；
7.  数据协方差或误差模型；
8.  有效性切割或能区说明。

这些条件并不要求两项分析使用相同坐标。若基变换可逆、输入方案的重展开已知，
且 RGE 与阈值匹配覆盖两者之间的尺度区间，就能把它们运输到共同坐标后组合。
反之，只有单系数切片、没有协方差或没有能区信息的结果，最多能作定性比较。

**研究边界 19.2.** 维数八全局 RGE、NLO SMEFT 的全通道自动化、完整 HEFT-to-data
接口和若干低能强子/核矩阵元仍依赖专门外部计算。使用这些结果时，应引用其具体
方案与适用能区，不能把一个方向中的完成度外推到其他 sector。

## 19.3 拆解一个现象学结果

给定一篇声称限制 Wilson 系数的结果，应按以下顺序拆解：

1.  它限制的是 SMEFT、HEFT、LEFT 还是某个 simplified model；
2.  Wilson 系数在哪个尺度和哪个基中给出；
3.  数据是低能、Z-pole、Higgs、top 还是高能 tail；
4.  flavor 是否 full、diagonal、universal 或 MFV；
5.  是否只开一个系数；
6.  是否运行和匹配；
7.  是否给出协方差；
8.  是否检查 EFT validity。

**例 19.3（高质量 dilepton tail）.** 若分析使用 $pp\to\ell^+\ell^-$ 高质量 bins 并限制 semileptonic 四费米子算符，则最少要给出 partonic energy proxy、最大 dilepton invariant mass、PDF 设置、是否保留 $[C^{(6)}]^2/\Lambda_{\rm ref}^4$、flavor 组合和 quark chirality。还要分别给出对 $M_{\rm gap}$ 条件化的局域切割与 Wilson 插入层级；缺少任一项，结果就不能直接与 flavor 或 LEP 约束合并。

## 19.4 SMEFT、HEFT、LEFT 的边界

**原则 19.4.** 高尺度新物理若尊重线性电弱实现，且低能自由度只含标准模型场，则优先用 SMEFT。若 Higgs 不作为严格 $SU(2)_L$ 双重态的线性成员组织，则应考虑 HEFT。若能量低于电弱尺度并积掉 $W,Z,h,t$，则应使用 LEFT。

**警告 19.5.** 同一篇现象学论文可能同时使用 SMEFT 和 LEFT。严格写法必须给出二者的匹配尺度，不能把 LEFT Wilson 系数直接称为 SMEFT Wilson 系数。

## 19.5 从通用接口到研究边界

各方向真正共享的是 EFT 定义、匹配、幂计数、RGE、EOM 商、flavor/CP 和输入参数
这些运输规则。Higgs、电弱、top 与高能散射主要在破缺相振幅和 collider likelihood
处分叉；flavor、EDM、$g-2$ 和 LNV 则继续经过 LEFT matching 与低能矩阵元。维数八
全基、完整全局拟合、NLO 自动化和非线性 HEFT 不是同一项“更高阶修正”，而是分别
扩展算符空间、统计接口、圈精度与对称性实现，必须逐项说明新增假设。

## 19.6 跨尺度研究的共同骨架

从重阈值到观测量的共同骨架可以写成：在 $\mu_{\rm match}$ 匹配 Wilson 初值，
在固定基与方案中运行，必要时跨电弱阈值改用 LEFT，最后以过程矩阵元和协方差映到
数据。不同方向的差别集中在这条链所需的外部输入以及可用能区，而不是 EFT 原理本身。
因此跨通道组合首先是接口相容性问题；只有这些接口对齐后，更多数据才真正增加
Wilson 空间中的独立约束方向。

## 练习

**练习 19.1.** 选取 Higgs signal strength 中一个通道，列出它至少依赖哪些 SMEFT 元数据。

**练习 19.2.** 解释为什么高 $p_T$ dilepton 约束需要显式报告有效性切割。

**练习 19.3.** 给出一个同时需要 SMEFT 和 LEFT 的 flavor 观测量，并标出匹配尺度。

**练习 19.4.** 对一个高质量 dilepton SMEFT 结果列出八项元数据检查。
