# 第三章：Equivariant sheaves、六函子与 perverse t-structure

## 本章目标

本章建立几何表示论的 sheaf-theoretic 语言：equivariant derived categories、constructible complexes、六函子、Verdier duality、perverse t-structure 和 intersection complexes。这里的重点是类型约定和可验证的基本命题，而不是重证 BBD 理论。

## 依赖前置知识

需要附录 A 的商栈和六函子模板，第一章的 flag variety 和 Schubert 分层。

## 3.1 Constructible derived categories

**约定 3.1.** 本章固定一个 sheaf theory 模型。若 $k=\mathbb C$，可使用 classical topology 上的 $E$-vector sheaves。若 $k$ 任意且 $\ell\ne\operatorname{char}k$，可使用 $\ell$-adic constructible sheaves。正文中不在同一证明里混用两种模型。

**定义 3.2.** 给定带有限分层
$$
X=\coprod_{\alpha\in A}X_\alpha
$$
的代数簇，$D^b_c(X,E)$ 表示这样的有界导出范畴：其对象 $\mathcal F$ 的 cohomology sheaves 在每个 stratum $X_\alpha$ 上为局部常值且有限秩。

**定义 3.3.** 若代数群 $H$ 作用于 $X$ 并保持分层，则
$$
D^b_H(X,E)=D^b_c([X/H],E)
$$
表示 $H$-equivariant constructible derived category。等价地，可把对象看作 $\mathcal F\in D^b_c(X,E)$ 连同作用图
$$
a:H\times X\to X,\qquad p:H\times X\to X
$$
上的 descent isomorphism
$$
a^\ast\mathcal F\simeq p^\ast\mathcal F
$$
并满足 cocycle condition。

**命题 3.4.** 若 $X=H/K$，则在附录 A 的假设下，$D^b_H(X,E)\simeq D^b(BK,E)$。

**证明.** 由命题 A.8 有 quotient stack 等价 $[H/K/H]\simeq BK$。对等价的 stacks 应用 constructible derived category functor 得到所需等价。若采用 descent datum 口径，则 $H$-equivariant sheaf on $H/K$ 被基点纤维和 $K$-monodromy 完全决定，得到同一结论。$\square$

## 3.2 六函子和 Verdier duality

**约定 3.5.** 对 morphism $f:X\to Y$，本书把
$$
f^\ast,\ f_\ast,\ f_!,\ f^!
$$
均视为导出 functor。未导出 functor 不使用相同符号。

**外部输入定理 3.6.** 在 constructible sheaf theory 的标准假设下，六函子满足 adjunctions
$$
f^\ast\dashv f_\ast,\qquad f_!\dashv f^!,
$$
proper base change、projection formula 和 Verdier duality compatibility。  
来源：BBD、Kashiwara-Schapira 或相应 l-adic six-functor formalism。

**定义 3.7.** Verdier duality 定义为
$$
\mathbb D_X(\mathcal F)=R\mathcal Hom(\mathcal F,\omega_X^\bullet).
$$
它是 contravariant functor
$$
\mathbb D_X:D^b_c(X,E)^{op}\to D^b_c(X,E).
$$

**命题 3.8.** 若 $f:X\to Y$ proper，则存在自然同构
$$
\mathbb D_Y f_\ast\simeq f_\ast \mathbb D_X.
$$

**证明.** 由 properness，$f_\ast\simeq f_!$。Verdier duality 的六函子相容性给出
$$
\mathbb D_Y f_!\simeq f_\ast\mathbb D_X.
$$
代入 $f_!=f_\ast$ 得到结论。该证明的实质输入是外部输入定理 3.6。$\square$

## 3.3 Perverse t-structure

**定义 3.9.** 假设 $X$ 为复代数簇或具有合适维数函数的 $k$-簇。middle perversity 的 aisle 和 co-aisle 由条件
$$
{}^pD^{\le0}(X)=
\{\mathcal F\mid \dim\{x\in X\mid H^i(i_x^\ast\mathcal F)\ne0\}\le -i,\ \forall i\}
$$
和
$$
{}^pD^{\ge0}(X)=
\{\mathcal F\mid \dim\{x\in X\mid H^i(i_x^!\mathcal F)\ne0\}\le i,\ \forall i\}
$$
刻画。其心记为
$$
\operatorname{Perv}(X,E)={}^pD^{\le0}(X)\cap{}^pD^{\ge0}(X).
$$

更常用的等价 stratum-wise 表述为：若 $j_\alpha:X_\alpha\hookrightarrow X$ 是维数 $d_\alpha$ 的 stratum，则要求
$$
H^i(j_\alpha^\ast\mathcal F)=0\quad(i>-d_\alpha),
$$
以及
$$
H^i(j_\alpha^!\mathcal F)=0\quad(i< -d_\alpha).
$$

**外部输入定理 3.10.** 上述条件定义 $D^b_c(X,E)$ 上的 t-structure，其心 $\operatorname{Perv}(X,E)$ 是 artinian and noetherian abelian category，在有限分层和有限系数条件下有有限长度。  
来源：BBD。

**命题 3.11.** 若 $X$ 光滑连通纯维数 $d$，则局部系统 $\mathcal L$ 的 shift $\mathcal L[d]$ 是 perverse sheaf。

**证明.** 对唯一光滑 stratum $X$，$j=\operatorname{id}_X$。有
$$
H^i(j^\ast\mathcal L[d])=H^{i+d}(\mathcal L),
$$
仅当 $i=-d$ 时可能非零，因此满足 $i>-d$ 的 vanishing。又因为 $X$ 光滑，$j^!=j^\ast$ 对 identity map 成立，所以 cosupport 条件同样满足。$\square$

**例 3.12.** 若 $X=\mathbb A^1$，则常值 sheaf $E_X[1]$ 是 perverse sheaf，而 $E_X$ 不是 perverse sheaf。原因是唯一 open stratum 维数为 $1$，perverse normalization 要求局部系统放在 cohomological degree $-1$。

**命题 3.13.** 对闭嵌入 $i:Z\hookrightarrow X$，若 $Z$ 光滑纯维数 $d$，则 $i_\ast E_Z[d]$ 是支撑在 $Z$ 上的 perverse sheaf，前提是 $i_\ast$ 对闭嵌入保持 perverse sheaves。

**证明.** 由命题 3.11，$E_Z[d]$ 在 $Z$ 上 perverse。闭嵌入满足 $i_\ast=i_!$，且 BBD formalism 中闭嵌入的 $i_\ast$ 是 t-exact 的外部输入。因此 $i_\ast E_Z[d]$ perverse。该命题的内部部分是 shift 检查；t-exactness 属于 perverse formalism。$\square$

## 3.4 Intersection complexes

**定义 3.14.** 令 $S\subset X$ 为光滑 locally closed stratum，$j:S\hookrightarrow\overline S$，$\mathcal L$ 为 $S$ 上的 local system，$d=\dim S$。intersection complex 定义为 middle extension
$$
\operatorname{IC}(\overline S,\mathcal L)=j_{!*}(\mathcal L[d]).
$$
若 $\mathcal L=E_S$，简记为 $\operatorname{IC}_{\overline S}$。

**外部输入定理 3.15.** middle extension $j_{!*}$ 存在且保持 perversity；$\operatorname{IC}(\overline S,\mathcal L)$ 是 simple perverse sheaf，前提是 $\mathcal L$ 是 irreducible local system。  
来源：BBD。

**定义 3.16.** 对 Schubert variety $\overline X_w\subset G/B$，本书采用 normalization
$$
\operatorname{IC}_w=\operatorname{IC}(\overline X_w,E_{X_w}).
$$
由于 $X_w\simeq\mathbb A^{\ell(w)}$，其常值 sheaf shift 为 $E_{X_w}[\ell(w)]$。

**命题 3.17.** 若 $\overline S$ 光滑且 $S$ 是其 dense open stratum，边界 codimension 至少为 $1$，则
$$
\operatorname{IC}(\overline S,E_S)\simeq E_{\overline S}[\dim S].
$$

**证明.** 因为 $\overline S$ 光滑连通纯维数 $\dim S$，命题 3.11 给出 $E_{\overline S}[\dim S]$ perverse。其限制到 $S$ 为 $E_S[\dim S]$。光滑常值 sheaf 没有支撑在边界上的 perverse subobject 或 quotient；这一点可由 local normal slice 的 perverse support/cosupport 条件验证，或作为 middle extension formalism 的标准结论。由 middle extension 的唯一性得到同构。$\square$

## 3.5 Decomposition theorem 的边界

**外部输入定理 3.18.** Decomposition theorem：对复代数簇的 proper map $f:X\to Y$，若 $X$ 光滑或更一般地取 semisimple perverse sheaf 输入，则 $Rf_\ast$ 分解为 shifted semisimple perverse sheaves 的直和，并满足相对 hard Lefschetz。  
用途：Springer theory、IC sheaves、KL positivity、geometric Satake。来源：BBD 或 Saito mixed Hodge modules 版本。

**警告 3.19.** 本书不会把 decomposition theorem 当作形式代数事实。它依赖深层几何，包括 hard Lefschetz、weights 或 Hodge theory。凡是使用它推出 semisimplicity、positivity 或 purity 的地方，必须标注外部输入。

**检查表 3.20.** 使用 decomposition theorem 前必须说明：

1. $f$ 是否 proper；
2. 输入对象是否 semisimple perverse；
3. 使用 Betti、l-adic 还是 mixed Hodge module 版本；
4. 是否需要相对 hard Lefschetz；
5. 是否使用 purity 推出系数非负。

## 本章小结

本章固定了 equivariant sheaves 的 quotient stack 口径、六函子符号、perverse t-structure 和 IC sheaf normalization。内部证明只覆盖齐性空间 equivariant sheaves、proper 情形下 Verdier duality 相容的形式推论和光滑簇上局部系统 shift 的 perversity。BBD formalism 和 decomposition theorem 均为外部输入。

## 练习

**练习 3.1.** 令 $X=\mathbb A^1$，分层为 $\{0\}\sqcup\mathbb G_m$。写出 perverse sheaf support/cosupport 条件在两个 stratum 上的形式。

**练习 3.2.** 对 $SL_2/B\simeq\mathbb P^1$ 的 Schubert 分层，写出 $\operatorname{IC}_e$ 和 $\operatorname{IC}_s$。

**练习 3.3.** 证明若 $f:X\to Y$ 是同构，则 $f^\ast,f_\ast,f_!,f^!$ 在 constructible derived categories 上均为等价，并说明这些等价与 Verdier duality 相容。
