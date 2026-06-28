# 附录 A：复几何定理的证明路线

## A.0 目标

第三卷多次把 Clausen-Scholze 的深层复几何定理作为输入。本附录把这些定理的证明路线集中写出，说明每一步依赖前两卷中的哪些工具。

## A.1 相干上同调有限性

**目标.** 对紧复流形 $X$ 和相干层 $\mathcal F$，证明

$$
\dim_{\mathbb C}H^i(X,\mathcal F)<\infty.
$$

**路线.**

1. 用局部分解把相干层归约到有限自由 $\mathcal O_X$-模的 cokernel。
2. 用 Dolbeault resolution 把 $R\Gamma(X,\mathcal F)$ 表示为全局微分形式复形。
3. 把微分形式空间放入 liquid 向量空间范畴。
4. 使用椭圆算子或 Fredholm 理论证明复形同调有限维。
5. 把有限维性翻译为 condensed/analytic 范畴中的 compact/perfect 性质。

**前两卷依赖.**

- 第二卷第五章：liquid 向量空间入口。
- 第二卷第七章：$f_!$ 和相干对偶入口。

## A.2 Serre 对偶

**目标.** 对紧 $n$ 维复流形 $X$，证明配对

$$
H^i(X,\mathcal F)\times
\operatorname{Ext}^{n-i}(\mathcal F,\omega_X)
\to\mathbb C
$$

完美。

**路线.**

1. 用 Dolbeault 复形表达两侧。
2. 构造积分配对
   $$
   \int_X \alpha\wedge\beta.
   $$
3. 检查配对与 $\bar\partial$ 兼容，因此下降到同调。
4. 用有限性定理把非退化配对升级为完美对偶。
5. 在范畴语言中识别为 $f_!\dashv f^!$ 的 trace/counit。

**关键检查.** 积分配对必须在 liquid/analytic 范畴中连续，并与导出 Hom 相容。

## A.3 GAGA

**目标.** 对 proper complex algebraic variety $X$，比较 algebraic 和 analytic 相干理论：

$$
D_{\operatorname{coh}}(X)
\simeq
D_{\operatorname{coh}}(X^{an}).
$$

**路线.**

1. 构造 analytification 函子
   $$
   \mathcal F\mapsto\mathcal F^{an}.
   $$
2. 在仿射局部或 Stein 局部证明相干对象比较。
3. 使用 properness 得到上同调有限性和下降。
4. 通过 Cech descent 粘合局部比较。
5. 证明导出全局截面与比较函子相容。

**关键检查.** 非 proper 情形一般不满足 GAGA；properness 是上同调控制和粘合的核心。

## A.4 Riemann-Roch

**目标.** 对紧复流形 $X$ 和向量丛 $E$，证明

$$
\chi(X,E)=\int_X\operatorname{ch}(E)\operatorname{td}(T_X).
$$

**路线.**

1. 用有限性定理定义 Euler characteristic。
2. 用 K-theory 表达向量丛或 perfect complexes 的类。
3. 构造 Chern character 和 Todd class。
4. 用 trace map 表达 Euler characteristic。
5. 证明 trace 与 characteristic classes 的 compatibility。

**关键检查.** condensed/analytic 框架提供的是同调代数环境；characteristic classes 的几何构造仍需要独立输入。

## A.5 六函子形式

**目标.** 把上述定理放入

$$
f^*,f_*,f_!,f^!,\otimes,R\mathcal Hom
$$

的形式中。

**路线.**

1. 对仿射和局部模型构造六函子。
2. 验证 base change、projection formula 和 adjunction。
3. 对一般空间用下降粘合。
4. 在 proper 情形比较 $f_!$ 与 $f_*$。
5. 用 $f^!$ 表达对偶层和 trace。

## A.6 本附录小结

第三卷的定理证明依赖三条主线：

1. Dolbeault/liquid 解析模型。
2. $f_!$、$f^!$ 和 trace 的相干对偶语言。
3. GAGA 与 characteristic classes 的几何比较。

本书给出路线和范畴翻译；完整证明仍以 Clausen-Scholze 讲义为准。
