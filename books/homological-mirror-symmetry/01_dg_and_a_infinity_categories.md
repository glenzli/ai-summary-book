# 第一章：dg 范畴、$A_\infty$ 范畴与预三角化

## 本章目标

本章建立 HMS 所需的增强范畴语言。重点是 dg category、$A_\infty$ category、quasi-equivalence、Morita equivalence、twisted complexes 和 pretriangulated envelope。后续的 Fukaya category 与 B-side derived category 都会用这些概念表达。

## 依赖前置知识

需要熟悉链复形、同调、张量积、普通范畴和函子。关于 signs 的完整
coderivation 公式与低阶展开见附录 B；本章的所有 $A_\infty$ 公式均按
该处 (B.1)--(B.3) 解读。

## 1.1 复形与 dg 范畴

**约定 1.1.** 本书采用 cohomological grading。一个复形 $C$ 是分次 $k$-向量空间 $C=\bigoplus_i C^i$ 和次数 $+1$ 映射 $d:C^i\to C^{i+1}$，满足 $d^2=0$。

**定义 1.2.** 一个 $k$-线性 dg category $\mathcal A$ 包括：

1. 一个对象类 $\operatorname{Ob}\mathcal A$；
2. 对任意对象 $X,Y$，一个复形 $\operatorname{hom}_{\mathcal A}(X,Y)$；
3. 对任意 $X,Y,Z$，次数 $0$ 的链映射
   $$
   \circ:\operatorname{hom}_{\mathcal A}(Y,Z)\otimes
   \operatorname{hom}_{\mathcal A}(X,Y)\to
   \operatorname{hom}_{\mathcal A}(X,Z);
   $$
4. 对任意 $X$，一个闭的零次元素 $1_X\in\operatorname{hom}_{\mathcal A}^0(X,X)$；

并满足严格结合律和单位律。链映射条件等价于齐次元素 $a:X\to Y$、$b:Y\to Z$ 满足
$$
d(b\circ a)=d(b)\circ a+(-1)^{|b|}b\circ d(a).
$$

**定义 1.3.** dg category $\mathcal A$ 的同伦范畴 $H^0(\mathcal A)$ 定义为：对象与 $\mathcal A$ 相同，morphism 集合为
$$
\operatorname{Hom}_{H^0(\mathcal A)}(X,Y)
=H^0\operatorname{hom}_{\mathcal A}(X,Y).
$$

**命题 1.4.** $H^0(\mathcal A)$ 是普通范畴。

**证明.** 由于 dg 复合是链映射，它把零次 cocycles 的复合送到零次 cocycles，并把 coboundary 诱导的等价关系送到 coboundary 诱导的等价关系。因此复合在 $H^0$ 上良定义。严格结合律和单位律在链级成立，取 $H^0$ 后仍成立。单位 $1_X$ 是闭元素，所以给出 $H^0$ 中的单位态射。证毕。

**例 1.5.** 设 $R$ 是 $k$-代数。复形的 dg category $\operatorname{Ch}(R)$ 以 cochain complexes of right $R$-modules 为对象，morphism complex 为 graded homomorphisms，微分为
$$
d(f)=d_N\circ f-(-1)^{|f|}f\circ d_M.
$$
命题 1.4 给出其同伦范畴 $H^0\operatorname{Ch}(R)$，其中 morphisms 是链映射模链同伦。

## 1.2 $A_\infty$ 范畴的 suspension 定义

dg category 的复合严格结合；Fukaya category 中的复合只在高阶同伦意义下结合。因此需要 $A_\infty$ category。

**定义 1.6.** 设 $\mathcal A$ 是带对象类的分次 $k$-线性数据，即对每对对象 $X,Y$ 有分次向量空间 $\operatorname{hom}_{\mathcal A}(X,Y)$。记
$$
s\operatorname{hom}_{\mathcal A}(X,Y)=\operatorname{hom}_{\mathcal A}(X,Y)[1].
$$
一个非弯曲 $A_\infty$ category 结构是在所有 composable 序列上给出次数 $+1$ 的 coderivation
$$
b:T^c(s\operatorname{hom}_{\mathcal A})\to T^c(s\operatorname{hom}_{\mathcal A})
$$
使得 $b^2=0$，其中 $T^c$ 是按对象匹配的 reduced tensor coalgebra，
coderivation 的延拓符号固定为公式 (B.1)。它的 Taylor components 记为
$$
b_d:
s\operatorname{hom}_{\mathcal A}(X_{d-1},X_d)\otimes\cdots\otimes
s\operatorname{hom}_{\mathcal A}(X_0,X_1)
\to s\operatorname{hom}_{\mathcal A}(X_0,X_d).
$$
desuspension 后得到次数 $2-d$ 的运算
$$
\mu^d:
\operatorname{hom}_{\mathcal A}(X_{d-1},X_d)\otimes\cdots\otimes
\operatorname{hom}_{\mathcal A}(X_0,X_1)
\to \operatorname{hom}_{\mathcal A}(X_0,X_d)[2-d].
$$

**解释 1.7.** 方程 $b^2=0$ 同时编码所有 $A_\infty$ 关系。其无省略符号
版本是命题 B.3，低阶关系 (B.4)--(B.6) 包括：

- $\mu^1$ 是微分；
- $\mu^2$ 与 $\mu^1$ 相容，即 $\mu^1$ 对 $\mu^2$ 满足带 Koszul 符号的 Leibniz 规则；
- $\mu^2$ 的结合律不要求链级严格成立，其失败由 $\mu^3$ 的边界控制；
- 更高 $\mu^d$ 控制更高结合同伦的相容性。

**定义 1.8.** 一个严格含单位 $A_\infty$ category 是带有元素
$e_X\in\operatorname{hom}_{\mathcal A}^0(X,X)$ 的 $A_\infty$ category。
写 $\epsilon_X=se_X$。对
$x\in s\operatorname{hom}_{\mathcal A}(X,Y)$，要求
$$
b_1(\epsilon_X)=0,\qquad
b_2(\epsilon_Y,x)=x,\qquad
b_2(x,\epsilon_X)=(-1)^{|x|+1}x,
$$
并且 $d\ne2$ 时，任何含 $\epsilon_X$ 输入的 $b_d$ 为零。按附录 B 的
desuspension 约定，这给出 $\mu^2(e_Y,a)=a=\mu^2(a,e_X)$。本书后续把
严格单位作为默认条件；若使用 cohomological 或 homotopy units，将显式
说明。

**命题 1.9.** 每个 dg category 给出一个严格含单位 $A_\infty$ category。

**证明.** 设 $\mathcal A$ 是 dg category。定义
$$
\mu^1=d,\qquad \mu^2(b,a)=b\circ a,\qquad \mu^d=0\quad(d\ge 3),
$$
并取 dg category 的单位作为 $A_\infty$ 单位。$A_\infty$ 方程在 $d=1$ 时是 $d^2=0$；在 $d=2$ 时是 dg 复合为链映射，即 Leibniz 规则；在 $d=3$ 时是复合的严格结合律；在 $d\ge4$ 时所有项都含有某个 $\mu^r$，$r\ge3$，或归约到已经检查的结合律与 Leibniz 规则。故得到严格含单位 $A_\infty$ category。证毕。

## 1.3 $A_\infty$ functor 与 quasi-equivalence

**定义 1.10.** 设 $\mathcal A,\mathcal B$ 是 $A_\infty$ categories。一个 $A_\infty$ functor
$$
F:\mathcal A\to\mathcal B
$$
包括对象映射 $X\mapsto FX$，以及次数 $1-d$ 的多线性映射
$$
F^d:
\operatorname{hom}_{\mathcal A}(X_{d-1},X_d)\otimes\cdots\otimes
\operatorname{hom}_{\mathcal A}(X_0,X_1)
\to
\operatorname{hom}_{\mathcal B}(FX_0,FX_d)[1-d],
$$
使得相应的 bar coalgebra morphism 与 coderivations 相容。若 $F^1$ 与单位相容且高阶分量在单位输入上退化，则称 $F$ 为严格含单位 functor。

**定义 1.11.** $A_\infty$ functor $F:\mathcal A\to\mathcal B$ 称为 quasi-equivalence，若：

1. 对任意 $X,Y\in\mathcal A$，链映射
   $$
   F^1:\operatorname{hom}_{\mathcal A}(X,Y)\to
   \operatorname{hom}_{\mathcal B}(FX,FY)
   $$
   是 quasi-isomorphism；
2. 诱导函子 $H^0(F):H^0(\mathcal A)\to H^0(\mathcal B)$ 本质满。

**引理 1.12.** quasi-equivalence 在 $H^0$ 上诱导普通范畴等价。

**证明.** 与命题 0.5 相同：第一条件给出 morphism 集合的双射，第二条件给出本质满。证毕。

**警告 1.13.** $H^0(F)$ 为等价不推出 $F$ 是 quasi-equivalence，因为前者只检测 morphism complexes 的零次上同调，不检测其他次数。

## 1.4 Modules、Yoneda 与 Morita 口径

**定义 1.14.** 设 $\mathcal A$ 是小 $A_\infty$ category。一个右 $\mathcal A$-module 是一个 $A_\infty$ functor
$$
\mathcal A^{op}\to \operatorname{Ch}_k
$$
的等价数据。所有右 modules 组成 dg 或 $A_\infty$ category，记为 $\operatorname{Mod}(\mathcal A)$。

**定义 1.15.** Yoneda module $Y_X$ 定义为
$$
Y_X(-)=\operatorname{hom}_{\mathcal A}(-,X).
$$
由 $X\mapsto Y_X$ 得到 Yoneda embedding
$$
Y:\mathcal A\to \operatorname{Mod}(\mathcal A).
$$

**外部输入定理 1.16（$A_\infty$ Yoneda）.** 对小、严格含单位
$A_\infty$ category，Yoneda embedding 是 cohomologically fully
faithful；即
$$
\operatorname{hom}_{\mathcal A}(X,Y)\to
\operatorname{hom}_{\operatorname{Mod}(\mathcal A)}(Y_X,Y_Y)
$$
是 quasi-isomorphism。  
来源：Lefevre-Hasegawa 的 $A_\infty$ categories 与 modules 理论；完整
输入边界与定位见外部输入定理 B.11 及 theorem locator。

**定义 1.17.** $\operatorname{Perf}(\mathcal A)$ 是 $\operatorname{Mod}(\mathcal A)$ 中由 representable modules 经过有限 cones、shifts、direct summands 和 quasi-isomorphism 闭包生成的 full subcategory。若函子 $F:\mathcal A\to\mathcal B$ 诱导
$$
\operatorname{Perf}(\mathcal A)\xrightarrow{\sim}\operatorname{Perf}(\mathcal B)
$$
的 quasi-equivalence，则称 $F$ 为 Morita equivalence。

**例 1.18.** 若 $\mathcal A$ 只有一个对象，且 endomorphism dg algebra 为 $A$，则 $\operatorname{Perf}(\mathcal A)$ 就是 perfect right dg $A$-modules 的范畴。此时 Morita equivalence 回到 dg algebras 的 derived Morita equivalence。

## 1.5 Twisted complexes 与预三角化

**定义 1.19.** 对严格含单位 $A_\infty$ category $\mathcal A$，twisted complex 是有限直和形式
$$
E=\bigoplus_i X_i[n_i]
$$
连同严格上三角的次数 $1$ endomorphism $\delta_E$。写
$\beta_E=s\delta_E$；本书的 Maurer--Cartan 方程是精确的 suspended 等式
$$
\sum_{d\ge1}b_d(\beta_E,\ldots,\beta_E)=0.
\tag{1.1}
$$
严格上三角性使该和有限。若改写成 unsuspended $\mu^d$，必须按附录 B
的 graded tensor-map convention 移动 suspensions；不能把 (1.1) 无符号地
替换为 $\sum_d\mu^d(\delta_E,\ldots,\delta_E)=0$。Twisted complexes
构成的 $A_\infty$ category 记为 $\operatorname{Tw}(\mathcal A)$。

**解释 1.20.** $\operatorname{Tw}(\mathcal A)$ 在 $\mathcal A$ 中形式添加 shifts 和 cones。它是把 $A_\infty$ category 变成适合三角结构的最小闭包之一。

**外部输入定理 1.21（有限 twisted-complex 预三角化包）.** 设 $k$ 是域，
$\mathcal A$ 是小、非弯曲、严格含单位的 $k$-线性 $A_\infty$ category。
则定义 1.19 的有限 twisted complexes 构成小、严格含单位、
pretriangulated $A_\infty$ category $\operatorname{Tw}(\mathcal A)$；
自然 embedding
$$
\mathcal A\longrightarrow\operatorname{Tw}(\mathcal A)
$$
在 morphism complexes 上 quasi-isomorphic。范畴
$H^0\operatorname{Tw}(\mathcal A)$ 带三角结构，其 shift 与 mapping-cone
triangles 由 twisted-complex 公式给出。本定理不添加任意 coproducts，也不
自动做 idempotent completion；后者属于 $\operatorname{Perf}(\mathcal A)$
口径。来源：Bondal--Kapranov、Keller、Lefevre-Hasegawa；在线定位见
`ONLINE_THEOREM_LOCATOR.md`。

**定义 1.22.** 若 Yoneda embedding 的像在 shifts 和 cones 下闭合，并且
$$
H^0(\mathcal A)\to H^0\operatorname{Tw}(\mathcal A)
$$
本质满，则称 $\mathcal A$ 为 pretriangulated。

## 1.6 与 HMS 的关系

HMS 中常见的几种等价强度如下。

**定义 1.23.** 设 $\mathcal A$ 是 A-side 的 $A_\infty$ category，$\mathcal B$ 是 B-side 的 dg category。

- quasi-equivalence 版本要求存在 $A_\infty$ functor $\mathcal A\to\mathcal B$，在 morphism complexes 上为 quasi-isomorphism 且在 $H^0$ 上本质满。
- split-generation 版本先在 A-side 找到对象集合 $\mathcal G$，证明其 split-generates $\mathcal A$，再识别 endomorphism algebra 与 B-side 的生成对象 endomorphism algebra。
- Morita 版本要求 $\operatorname{Perf}(\mathcal A)\simeq\operatorname{Perf}(\mathcal B)$。

**定义 1.23A（split-generation 的 Morita 口径）.** 设 $\mathcal A$ 是小、
严格含单位 $A_\infty$ category，$\mathcal G\subset\operatorname{Ob}\mathcal A$，
$\mathcal A_{\mathcal G}$ 是其张成的 full subcategory。称 $\mathcal G$
split-generates $\mathcal A$，若 representables $Y_G$（$G\in\mathcal G$）
的最小厚子范畴等于
$$
H^0\operatorname{Perf}(\mathcal A),
$$
等价地，inclusion $\mathcal A_{\mathcal G}\hookrightarrow\mathcal A$ 是
Morita equivalence。若 $\mathcal A$ 已 pretriangulated、idempotent-complete
且 Yoneda 给出 $\mathcal A\simeq\operatorname{Perf}(\mathcal A)$，这才等价于
“$\mathcal G$ 的厚闭包等于 $H^0(\mathcal A)$”。

**命题 1.24.** 设 $\mathcal A,\mathcal B$ 是小、严格含单位 dg 或
$A_\infty$ categories。若 $\mathcal G=\{G_i\}$ split-generates
$\mathcal A$，$\mathcal H=\{H_i\}$ split-generates $\mathcal B$，且 full subcategories
$$
\mathcal A_{\mathcal G}\subset\mathcal A,\qquad
\mathcal B_{\mathcal H}\subset\mathcal B
$$
quasi-equivalent，则 $\mathcal A$ 与 $\mathcal B$ Morita equivalent。

**证明.** split-generation 的含义是 representable modules $Y_{G_i}$ 经过 shifts、cones 和 direct summands 生成 $\operatorname{Perf}(\mathcal A)$，而 $Y_{H_i}$ 同样生成 $\operatorname{Perf}(\mathcal B)$。full subcategories 的 quasi-equivalence 诱导其 perfect module categories 的 quasi-equivalence。由于两边的 perfect categories 分别由这些生成对象的 representables 的厚闭包给出，诱导函子在厚闭包上本质满且在 morphism complexes 上保持 quasi-isomorphism。因此得到 Morita equivalence。证毕。

## 本章小结

dg category 是复形富化的范畴，$A_\infty$ category 是把结合律替换为一族高阶同伦的结构。HMS 必须在这些增强层面表述，因为 Fukaya categories 的高阶复合和 B-side 的导出增强都不是三角影子能完整记录的。Morita 口径允许通过生成对象和 endomorphism algebras 来证明 HMS。

## 练习

**练习 1.1.** 证明 dg category 的单位在 $H^0$ 中给出普通范畴单位。

**练习 1.2.** 对一个 dg algebra $A$，把它看成单对象 dg category，并写出 $\operatorname{Perf}(A)$ 的对象。

**练习 1.3.** 展开 $A_\infty$ 方程的 $d=1,2,3$ 情形，并说明 $\mu^3$ 如何测量 $\mu^2$ 的结合失败。

**练习 1.4.** 给出一个 quasi-equivalence 必然诱导 Morita equivalence 的证明。
