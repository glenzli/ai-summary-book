# 附录 L：Gross-Hopkins duality 与 Picard group 约定表

## L.1 对偶对象

| 符号 | 本书含义 | 风险 |
| --- | --- | --- |
| $I$ | Brown-Comenetz dualizing spectrum | 与 invariant ideal 不混淆 |
| $I_{\mathbb Z}$ | Anderson dualizing spectrum | 不等于 Brown-Comenetz dual |
| $I_n^{GH}$ | 第 $n$ 高度 Gross-Hopkins/Brown-Comenetz dualizing object | 悬挂 convention 依赖文献 |
| $S\langle\det\rangle$ | determinant sphere 或 determinant twist | 构造和归一化需定位 |

**约定 L.1.** 本书默认 $I_n^{GH}$ 表示第十章定义的 monochromatic Brown-Comenetz dualizing object。若引用文献用 $I_n$，正文必须在引用点写出对应关系。

## L.2 Gross-Hopkins duality 的公式模板

**模板 L.2.** Gross-Hopkins duality 的可引用公式必须写成：
$$
I_n^{GH}\simeq \Sigma^{a(n,p,\mathrm{conv})}S\langle\det\rangle\otimes P
$$
其中：

1. $a(n,p,\mathrm{conv})$ 是依赖 convention 的整数；
2. $S\langle\det\rangle$ 是 determinant twist；
3. $P$ 可能是 exotic Picard element；
4. 等价发生在 $\mathbf{Sp}_{K(n)}$；
5. 需要说明素数、高度和参考文献。

**警告 L.3.** 在未确定 $P$ 是否 trivial 前，不得把 $I_n^{GH}$ 简化为单纯悬挂的 determinant sphere。

## L.3 Picard group 分解

**定义 L.4.** Picard comparison map
$$
\operatorname{Pic}_{K(n)}\to \operatorname{Pic}_{\mathbb G_n}((E_n)_*)
$$
的 kernel 记作 $\kappa_n$，称为 exotic Picard subgroup。

**外部输入 L.5.** 在大素数或特定低高度情形，$\operatorname{Pic}_{K(n)}$ 可分解为悬挂部分、代数 character/determinant 部分和 exotic 部分。具体分解需按 Hopkins-Mahowald-Sadofsky、Hovey-Sadofsky、Goerss-Henn-Mahowald-Rezk、Mor 等结果定位。

**警告 L.6.** “大素数”通常意为 $p$ 相对 $n$ 足够大，例如 $2p-2>n^2$ 或类似范围；具体不等式随定理而变。

## L.4 低高度表

| 高度 | 典型现象 | 当前状态 |
| --- | --- | --- |
| $n=0$ | 有理稳定同伦，对偶退化为线性代数型 | 可内部处理 |
| $n=1$ | 与 $p$-adic K-theory、Adams operations、$J$ 相关 | 需低高度 locator |
| $n=2,p=3$ | Picard group 有经典计算，exotic 部分非平凡 | 需 GHMR locator |
| $n=2,p=2$ | Morava stabilizer cohomology 和 duality resolution 复杂 | 需 2022 计算 locator |
| 大素数 | Picard group 更接近代数描述 | 需 Hovey-Sadofsky/HMS locator |

## L.5 使用检查

**检查表 L.7.** 使用 Gross-Hopkins 或 Picard 结果前，必须记录：

1. $n,p$；
2. $I_n^{GH}$ 的定义；
3. determinant sphere 的 convention；
4. 是否存在 exotic factor；
5. comparison map 的目标；
6. 使用的 descent spectral sequence；
7. 是否只计算 $\pi_0$ 还是整个 Picard spectrum。

## 本附录小结

Gross-Hopkins duality 和 Picard group 是 convention-sensitive 的主题。正式教材中宁可保留完整模板，也不能写未定位的漂亮公式。
