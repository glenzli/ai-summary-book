# 第十一章：Higgs、电弱与 top 的 SMEFT 入口

## 本章目标

本章说明 SMEFT 如何进入 Higgs、电弱精密和 top 物理。目标不是给出完整拟合，而是固定可观测量与 Wilson 系数连接的严格语言。

## 依赖前置知识

需要第六至八章的 SMEFT 定义、Warsaw basis 和观测量展开。

## 11.1 Higgs 耦合

**定义 11.1（Higgs 信号强度）.** 对产生道 $i$ 和衰变道 $f$，实验常报告
$$
\mu_{if}
=
\frac{\sigma_i\,\mathrm{BR}_f}{(\sigma_i\,\mathrm{BR}_f)_{\mathrm{SM}}}.
$$

**解释 11.2.** SMEFT 修正可进入产生截面、部分宽度、总宽度和接受度。若只把 $\mu_{if}$ 解释为单个耦合缩放，通常隐含了强假设。

**例 11.3（$H^\dagger HG_{\mu\nu}G^{\mu\nu}$）.** 算符
$$
\mathcal O_{HG}=H^\dagger H\,G_{\mu\nu}^A G^{A\mu\nu}
$$
在电弱破缺后含
$$
v h\,G_{\mu\nu}^A G^{A\mu\nu}
$$
项，直接修正 $gg\to h$ 和 $h\to gg$。

## 11.2 信号强度的线性化

令
$$
\sigma_i=\sigma_i^{\rm SM}(1+\delta_i),\qquad
\Gamma_f=\Gamma_f^{\rm SM}(1+\delta_f),
$$
并令总宽度
$$
\Gamma_{\rm tot}
=\Gamma_{\rm tot}^{\rm SM}(1+\delta_{\rm tot}).
$$
则
$$
\mathrm{BR}_f
={\Gamma_f\over\Gamma_{\rm tot}}
=\mathrm{BR}_f^{\rm SM}(1+\delta_f-\delta_{\rm tot})
+O(\delta^2),
$$
所以
$$
\mu_{if}=1+\delta_i+\delta_f-\delta_{\rm tot}+O(\delta^2).
$$

**命题 11.4（Higgs 信号强度的退化）.** 单个 $\mu_{if}$ 只能约束组合
$$
\delta_i+\delta_f-\delta_{\rm tot}.
$$
因此除非引入额外假设，它不能单独确定产生、衰变和总宽度中的 Wilson 系数。

**证明.** 将上述线性展开代入定义 11.1。$\square$

## 11.3 电弱精密

**定义 11.5（oblique 口径）.** 电弱精密观测可部分组织为规范玻色子两点函数的修正，常用 $S,T,U$ 或 SMEFT Wilson 系数组合描述。

**警告 11.6.** Oblique 参数是特定假设下的低维投影。一般 SMEFT 还含顶点修正和四费米子接触项，不应把 oblique fit 当作完整 SMEFT fit。

在 Warsaw basis 中，电弱精密数据常对以下结构敏感：
$$
{\cal O}_{HD},\quad {\cal O}_{HWB},\quad
{\cal O}_{H\ell}^{(1,3)},\quad
{\cal O}_{He},\quad
{\cal O}_{Hq}^{(1,3)},\quad
{\cal O}_{Hu},\quad
{\cal O}_{Hd},\quad
{\cal O}_{\ell\ell}.
$$
其中 ${\cal O}_{HD}$ 和 ${\cal O}_{HWB}$ 改变中性规范玻色子质量和 kinetic mixing，current 算符改变 $Zf\bar f$ 或 $Wf\bar f'$ 顶点，四轻子算符进入 $G_F$ 的抽取。

## 11.4 top 物理

**解释 11.7.** top 物理对 SMEFT 特别敏感，因为 top Yukawa 大，LHC 可探测高能尾部。常见算符包括 top-Higgs Yukawa 修正、top-gluon dipole、四费米子 top 算符和电弱 top current 修正。

**例 11.8（top-gluon dipole）.** Warsaw 算符
$$
{\cal O}_{uG}^{pr}=(\bar q_p\sigma^{\mu\nu}T^Au_r)\widetilde H\,G_{\mu\nu}^A
$$
在 $p=r=3$ 且电弱破缺后给出
$$
{v\over\sqrt2}(\bar t_L\sigma^{\mu\nu}T^At_R)G_{\mu\nu}^A
$$
以及含一个 Higgs 的 $t\bar tgh$ contact。它同时影响 $t\bar t$ 产生和 $t\bar th$ 过程，因此单过程拟合容易产生退化。

**原则 11.9（高能尾部纪律）.** 对 top 和 diboson 的高能分布，必须同时报告 EFT 截断、能量切割和维数六平方项处理方式。

## 本章小结

Higgs、电弱和 top 数据是 SMEFT 全局分析核心来源，但每个观测量到 Wilson 系数的映射都依赖基、输入参数方案和截断选择。

## 练习

**练习 11.1.** 展开 $H=(0,(v+h)/\sqrt2)^T$，求 $\mathcal O_{HG}$ 中含一个 Higgs 场的项。

**练习 11.2.** 说明为什么只用 Higgs 信号强度无法唯一确定所有 SMEFT Higgs 算符系数。

**练习 11.3.** 由 $\mu_{if}=1+\delta_i+\delta_f-\delta_{\rm tot}$ 说明为什么总宽度假设会影响 Higgs Wilson 系数限制。
