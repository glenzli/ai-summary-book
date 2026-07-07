# 第二十二章：可观测量到算符的图谱

## 本章目标

本章把第十九章的方向地图细化为 observables-to-operators 图谱。它给出每类分析最先检查的 Warsaw 算符族、常见退化方向和必须报告的元数据。

## 依赖前置知识

需要第十三章的算符表、第十六章的破缺相输入方案、第十七章的发布标准和第二十一章的输入方案线性代数。

## 22.1 电弱精密

| 观测量族 | 领先算符族 | 主要退化 | 必报元数据 |
| --- | --- | --- | --- |
| $m_W$ | ${\cal O}_{HD},{\cal O}_{HWB},{\cal O}_{H\ell}^{(3)},{\cal O}_{\ell\ell}$ | 输入方案退化 | 输入集、muon-decay 约定、线性/平方截断 |
| $Zf\bar f$ partial widths | ${\cal O}_{Hf}^{(1,3)}$、${\cal O}_{HD}$、${\cal O}_{HWB}$ | universal vs nonuniversal | flavor 假设、LEP/SLC 协方差 |
| asymmetries | 同上 | left/right coupling 组合 | pseudo-observable 定义 |
| triple gauge coupling | ${\cal O}_W,{\cal O}_{HWB},{\cal O}_{HW},{\cal O}_{HB}$ | diboson 与 Higgs 相关 | anomalous-coupling 规范化 |

## 22.2 Higgs

| 通道 | 领先算符族 | 直接效应 | 共同拟合风险 |
| --- | --- | --- | --- |
| gluon fusion | ${\cal O}_{HG}$、top Yukawa-like ${\cal O}_{uH}$ | $hG_{\mu\nu}G^{\mu\nu}$ contact 与 top loop 改变 | 与 $t\bar th$、Higgs decay 退化 |
| $h\to\gamma\gamma$ | ${\cal O}_{HW},{\cal O}_{HB},{\cal O}_{HWB}$、dipoles | $hF_{\mu\nu}F^{\mu\nu}$ contact | 输入方案和 SM loop 干涉 |
| $h\to ZZ^\ast,WW^\ast$ | ${\cal O}_{HD},{\cal O}_{H\Box},{\cal O}_{HW},{\cal O}_{HWB}$ | vertex、kinetic 和 width 改变 | 与 EWPO 强相关 |
| Yukawa channels | ${\cal O}_{eH},{\cal O}_{uH},{\cal O}_{dH}$ | fermion mass 与 Yukawa 同时重定义 | mass scheme |

## 22.3 Top 与高能散射

| 观测量族 | 领先算符族 | 有效性问题 |
| --- | --- | --- |
| $t\bar t$ inclusive | ${\cal O}_{uG}$、四夸克算符 | PDF、scale、quadratic terms |
| $t\bar t$ differential high-$p_T$ | 四夸克、dipole、current 算符 | $E/\Lambda$ 增强，必须切 bin |
| single top | ${\cal O}_{Hq}^{(3)}$、${\cal O}_{Hud}$、dipoles | charged-current flavor 假设 |
| high-mass dilepton | semileptonic 四费米子 | contact term 随 $s/\Lambda^2$ 增长 |
| VBS/diboson tails | gauge-Higgs、$X^3$、dimension-eight | 维数六平方和维数八竞争 |

## 22.4 Flavor、EDM 与低能

| 方向 | SMEFT 算符族 | 低能接口 | 主要不确定性 |
| --- | --- | --- | --- |
| $b\to s\ell^+\ell^-$ | semileptonic 四费米子、current 算符 | LEFT/WET Wilson 系数 | hadronic matrix elements |
| charged-current anomalies | ${\cal O}_{\ell q}^{(3)}$、${\cal O}_{Hq}^{(3)}$ | beta decay、meson decay、tau decay | CKM 和 flavor 假设 |
| LFV decays | lepton dipoles、four-lepton、semileptonic | low-energy decay rates | flavor off-diagonal bounds |
| EDM | CP-odd bosonic、fermion dipoles、four-fermion | hadronic/nuclear matrix elements | RGE mixing、matrix elements |
| $(g-2)_\ell$ | lepton dipole、semileptonic tensor | QED/QCD running | chirality enhancement |
| $0\nu\beta\beta$ | Weinberg operator、LNV higher-dimensional operators | nuclear EFT | Majorana phases、nuclear matrix elements |

## 22.5 全局拟合矩阵

一个正式全局拟合可抽象为
$$
d_a=t_a^{\rm SM}+M_{ai}C_i+Q_{aij}C_iC_j+\eta_a,
$$
其中 $d_a$ 是数据，$M_{ai}$ 是线性 SMEFT 响应，$Q_{aij}$ 是维数六平方响应，$\eta_a$ 包含理论和实验误差。

**规则 22.1（图谱到拟合）.** 任何一行 observables-to-operators 图谱进入拟合前，必须补齐：

1.  Wilson 系数定义尺度；
2.  RGE 是否使用；
3.  输入参数方案；
4.  flavor 和 CP 口径；
5.  covariance matrix；
6.  EFT validity cut；
7.  是否保留 $Q_{aij}$。

## 22.6 Worked map：high-mass dilepton

考虑 $pp\to\ell^+\ell^-$ 的高质量尾部。在 parton 层面，semileptonic contact operator 给出
$$
{\cal A}_{\rm EFT}\sim {C_{\ell q}\over\Lambda^2}s,
$$
而 SM Drell-Yan 振幅由 $\gamma/Z$ 交换给出。线性干涉对截面的贡献尺度为
$$
{\Delta\sigma\over\sigma_{\rm SM}}
\sim
{C_{\ell q}s\over\Lambda^2}
\times(\hbox{chirality and PDF weights}).
$$
因此高 $m_{\ell\ell}$ bin 灵敏度强，但 EFT validity 风险也强。

正式使用该图谱时必须指定：

1.  lepton flavor：$e,\mu,\tau$ 或组合；
2.  quark flavor 和 chirality；
3.  invariant-mass bin；
4.  PDF 和 QCD order；
5.  $\Lambda$ 与最大 partonic energy 的关系；
6.  是否保留 quadratic term；
7.  是否与 LEP 或 low-energy 数据联合。

**结论 22.2.** 该例展示了本章表格的用法：表格只给起点，正式分析还必须把算符族投影到外态、flavor、能区和 likelihood。

## 本章小结

SMEFT 现象学的核心不是把一个可观测量对应到一个算符，而是构造从算符空间到数据空间的线性或二次映射。本章给出的是正式分析前的最低图谱。

## 练习

**练习 22.1.** 说明为什么 $h\to\gamma\gamma$ 不能只约束一个 Wilson 系数。

**练习 22.2.** 对 high-mass dilepton 约束，写出需要报告的 EFT validity 信息。

**练习 22.3.** 选一个 flavor 观测量，说明 SMEFT 到 LEFT 匹配为何不可省略。

**练习 22.4.** 对 high-mass dilepton worked map，解释为什么灵敏度和有效性风险都随 $s$ 增大。
