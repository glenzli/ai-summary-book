# 第十四章：Flavor、CP 与 Wilson 系数空间

## 本章目标

本章解释为什么 Warsaw basis 的 59 个结构不是实际拟合参数个数，并给出 flavor、Hermiticity、CP 和最小 flavor violation 的基本纪律。

## 依赖前置知识

需要第十三章的算符表和标准模型 Yukawa 结构。

## 14.1 Flavor 指标

**定义 14.1（Wilson flavor tensor）.** 若算符 $\mathcal O_i$ 带 flavor 指标，例如
$$
\mathcal O_{Hq}^{(1)pr},
$$
则其 Wilson 系数也是 flavor 张量
$$
C_{Hq}^{(1)pr}.
$$
四费米子算符的系数通常为四指标张量 $C^{prst}$。

**警告 14.2.** “打开一个算符”与“打开一个 Wilson 参数”不是同义语。一个算符结构可能对应多个 flavor 分量。

## 14.2 Hermiticity

**命题 14.3（Hermitian current 算符的系数约束）.** 若算符满足
$$
(\mathcal O^{pr})^\dagger=\mathcal O^{rp},
$$
则拉氏量实性要求
$$
C^{pr}=(C^{rp})^\ast.
$$

**证明（书内推导）.** 拉氏量中对应项为
$$
C^{pr}\mathcal O^{pr}.
$$
取 Hermitian conjugate 得
$$
(C^{pr})^\ast\mathcal O^{rp}.
$$
要求总和在交换 dummy flavor 指标后不变，即得 $C^{pr}=(C^{rp})^\ast$。$\square$

## 14.3 CP 分类

**定义 14.4（CP-even 与 CP-odd 系数）.** 若某算符在 CP 下变号，则其实 Wilson 系数的非零值会产生 CP violation；若算符 CP-even，则实系数通常保持 CP，复相或 flavor 结构仍可能引入 CP violation。

**例 14.5.** 含 dual field strength 的
$$
\mathcal O_{H\widetilde G},\quad
\mathcal O_{H\widetilde W},\quad
\mathcal O_{H\widetilde B},\quad
\mathcal O_{H\widetilde WB}
$$
是 CP-odd 结构的标准例子。

## 14.4 常用 flavor 口径表

| 口径 | 参数空间含义 | 适用场景 | 风险 |
| --- | --- | --- | --- |
| flavor universal | 同类三代共享一个系数 | 粗略高能拟合、universal new physics | 会漏掉 flavor violation |
| diagonal but nonuniversal | 只开 $p=r$，不同代不同 | top/Higgs/lepton universality 测试 | 忽略 flavor-changing neutral current |
| full flavor | 保留所有 $p,r,s,t$ | flavor 物理、UV 匹配 | 参数数目巨大，需 Hermiticity 和交换对称 |
| MFV | Wilson 系数由 Yukawa spurion 展开 | 抑制 FCNC 的模型化分析 | 是强假设，不是模型无关结论 |
| CP conserving | 只保留 CP-even 或实系数子空间 | EDM 不敏感或假设无新 CP 相 | 会人为去掉 CP-odd 约束 |

**例 14.6（三代 Hermitian 二指标系数）.** 若 $C^{pr}$ 满足
$$
C^{pr}=(C^{rp})^\ast,
$$
则 $C$ 是 $3\times 3$ Hermitian 矩阵。它含有 $3$ 个实对角元和 $3$ 个复非对角元，因此共有
$$
3+2\binom32=9
$$
个实参数。若再要求 CP conservation 并取实对称矩阵，则参数数为
$$
3+\binom32=6.
$$

## 14.5 MFV

**定义 14.7（minimal flavor violation, MFV）.** MFV 假设 flavor violation 的唯一 spurion 来源为标准模型 Yukawa 矩阵。即在 flavor 对称群下，把 $Y_u,Y_d,Y_e$ 当作 spurion，使 Wilson 系数由这些 spurion 的组合构成。

**使用边界 14.8.** MFV 是强物理假设，不是规范对称性的结果。它可降低参数数目并抑制 flavor-changing neutral currents，但可能排除真实 UV 模型。

## 14.6 拟合空间

**原则 14.9（Wilson 空间报告）.** 一个 SMEFT 拟合必须报告：

1.  选用的算符基；
2.  flavor 假设；
3.  CP 假设；
4.  是否保留 flavor off-diagonal 分量；
5.  是否使用 Hermiticity 约束；
6.  参数在何尺度定义。

## 本章小结

SMEFT 的参数空间不是“59 维”。59 是结构目录；真实参数空间由 flavor、CP、Hermiticity 和分析假设共同决定。

## 练习

**练习 14.1.** 对 $C_{Hq}^{(1)pr}$ 写出三代 flavor 下 Hermitian 矩阵的实参数个数。

**练习 14.2.** 解释 MFV 为什么允许 flavor violation 但限制其结构。

**练习 14.3.** 比较 full flavor 与 diagonal nonuniversal 口径下 $C_{H\ell}^{(1)pr}$ 的实参数数目。
