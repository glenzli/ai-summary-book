# 第十二章：维数八、unitarity 与有效性边界

## 本章目标

本章固定 SMEFT 截断的边界问题：维数八算符、unitarity、positivity 和高能数据的有效性判断。

## 依赖前置知识

需要第六章的截断、第八章的观测量展开。

## 12.1 维数八为什么重要

**事实 12.1（同阶问题）.** 在观测量展开中，维数六平方项和维数八干涉项同为 $1/\Lambda^4$ 阶：
$$
|A_{\mathrm{SM}}+\Lambda^{-2}A_6+\Lambda^{-4}A_8|^2
=
|A_{\mathrm{SM}}|^2
+2\Lambda^{-2}\mathrm{Re}(A_{\mathrm{SM}}A_6^\ast)
+\Lambda^{-4}\left(|A_6|^2+2\mathrm{Re}(A_{\mathrm{SM}}A_8^\ast)\right)
+\cdots.
$$

**证明（书内推导）.** 直接展开振幅平方并按 $\Lambda^{-1}$ 次数收集。$\square$

**外部输入 12.2（维数八基）.** 完整 dimension-eight SMEFT operator basis 已有系统分类，但其规模远大于 dimension-six。本书将其作为研究边界，不在第一版逐项展开。

## 12.2 Perturbative unitarity

**定义 12.3（partial-wave unitarity 估计）.** 对 $2\to2$ 振幅，若某 partial wave $a_\ell(s)$ 满足
$$
|a_\ell(s)|\le 1,
$$
则称该能区未明显违反微扰 unitarity。实际分析常用更强的 $|\mathrm{Re}\,a_\ell|\le 1/2$ 作为保守条件。

**解释 12.4.** 高维算符振幅常随能量增长，如 $A_6\sim C E^2/\Lambda^2$。当该增长使 partial wave 接近 unitarity 界时，截断 EFT 已不能单独可信。

**例 12.5（常数角分布估计）.** 若某 $2\to2$ 振幅近似为
$$
{\cal A}(s,\cos\theta)=C{s\over\Lambda^2},
$$
则
$$
a_0(s)={1\over32\pi}\int_{-1}^{1}d\cos\theta\,{\cal A}
={C s\over16\pi\Lambda^2}.
$$
保守条件 $|{\rm Re}\,a_0|\le1/2$ 给出
$$
s\le {8\pi\Lambda^2\over |C|}.
$$
这不是新物理质量的精确界，而是 EFT 截断的自洽性警告。

## 12.3 Positivity 边界

**外部输入 12.6（positivity bounds）.** 在满足 Lorentz invariance、unitarity、analyticity、crossing symmetry 和适当 UV 行为的理论中，低能 Wilson 系数可受 positivity bounds 约束。

**使用边界.** Positivity 约束通常对某些 dimension-eight 组合更直接。将其用于 LHC 数据解释时，必须说明假设条件和过程能区。

**形式例 12.7（前向极限）.** 若弹性振幅在前向极限可写为
$$
{\cal A}(s,0)=a_0+a_1s+a_2s^2+\cdots,
$$
且满足适当解析性和 UV 有界性，则 dispersion relation 常推出 $a_2>0$ 类型的约束。EFT 中 $a_2$ 往往对应维数八 Wilson 系数组合。

## 12.4 有效性报告标准

**规则 12.8（发布 SMEFT 限制的最小信息）.** 一个 SMEFT 限制应报告：

1.  算符基；
2.  flavor 假设；
3.  截断阶数；
4.  是否包含维数六平方项；
5.  数据能区或最大不变量质量；
6.  输入参数方案；
7.  理论误差处理；
8.  是否使用 RG running。

## 12.5 截断方案的三种口径

| 口径 | 保留项 | 优点 | 风险 |
| --- | --- | --- | --- |
| 线性维数六 | SM 与 $A_6$ 干涉 | EFT 阶数清楚 | 干涉抑制时无灵敏度 |
| 维数六平方 | 再加 $|A_6|^2$ | 数值上常更稳定 | 与维数八同阶，不完整 |
| 到 $1/\Lambda^4$ 完整 | 加 $A_8$ 干涉 | 阶数一致 | 需要 dimension-eight basis 和更多系数 |

**原则 12.9.** 若使用维数六平方项，应把结果标为“dimension-six squared included”，不得称为“完整到 $1/\Lambda^4$”。

## 本章小结

SMEFT 的高能敏感性是优势也是风险。维数八、unitarity 和 positivity 不是附属细节，而是判断 EFT 解释是否可信的边界条件。

## 练习

**练习 12.1.** 对 $A=A_{\mathrm{SM}}+\Lambda^{-2}A_6+\Lambda^{-4}A_8$ 展开 $|A|^2$ 到 $\Lambda^{-4}$。

**练习 12.2.** 若某算符给出振幅 $A=C s/\Lambda^2$，估计何时可能接近 partial-wave unitarity 边界。

**练习 12.3.** 解释 positivity bound 为什么通常更自然地约束维数八而不是维数六。
