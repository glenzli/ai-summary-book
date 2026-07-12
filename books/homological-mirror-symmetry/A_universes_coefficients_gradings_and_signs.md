# 附录 A：集合论宇宙、系数、分次与符号约定

## A.1 宇宙

**约定 A.1.** 全书固定
$$
\mathcal U\in\mathcal V\in\mathcal W.
$$
对象集合默认 $\mathcal U$-小，范畴默认 $\mathcal V$-小或局部 $\mathcal U$-小。若使用 categories of categories，则提升到 $\mathcal W$。

## A.2 系数

**约定 A.2.** 基域为 $k$。exact 或 purely algebraic 章节默认 $k$-线性。非 exact Floer theory 默认使用 Novikov field $\Lambda$ 或 Novikov ring $\Lambda_{\ge0}$。

**定义 A.3.** 若 $A$ 为 $k$-线性 category，base change 到 $\Lambda$ 记为
$$
A_\Lambda=A\otimes_k\Lambda.
$$

## A.3 分次

**约定 A.4.** 复形采用 cohomological grading，微分次数为 $+1$。Shift 约定为
$$
C[1]^i=C^{i+1}.
$$

**约定 A.5.** $A_\infty$ 运算 $\mu^d$ 的次数为 $2-d$：
$$
\mu^d:\operatorname{hom}(X_{d-1},X_d)\otimes\cdots\otimes\operatorname{hom}(X_0,X_1)
\to\operatorname{hom}(X_0,X_d)[2-d].
$$

## A.4 符号

**约定 A.6.** 本书使用附录 B 的 suspension coalgebra 约定：
$|sa|=|a|-1$，张量次序为 $sa_d\otimes\cdots\otimes sa_1$，coderivation
延拓为 (B.1)，全部 Stasheff 恒等式为 (B.3)。低阶或 curved 公式不得再用
未定义的“$\pm$”替代这些符号。

**警告 A.7.** 不同 HMS 文献可能使用 homological grading、opposite order composition 或不同 suspension convention。引用外部公式时必须先翻译到本书约定。

## A.5 Strict、cohomological 与 curved units

**定义 A.8.** Strict unit 是链级单位；cohomological unit 只在 $H^0$ 或 cohomology category 中给出单位；curved unit 语境中 curvature 可为标量乘单位。

**约定 A.9.** 除非另有说明，本书 $A_\infty$ categories 默认严格含单位。Fukaya category 构造若只得到 cohomological unit，则必须使用 strictification 外部输入。

## 本附录小结

本附录固定全书小性、系数、分次和符号。HMS 文献的符号差异很大，后续 theorem locator 必须同时记录外部文献的 convention 与本书 convention 的翻译。
