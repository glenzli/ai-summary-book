# 第四十三章：Condensed sets、Solid modules 与解析范畴

## 本章目标

本章介绍 condensed mathematics 的范畴论骨架。Condensed sets 把拓扑信息编码为 profinite 测试对象上的 sheaves，从而把拓扑群、拓扑向量空间和解析对象放入 Grothendieck topos 与稳定同调代数的框架中。Solid modules 进一步给出适合泛函分析和解析几何的张量范畴。

## 依赖前置知识

需要站点、sheaf、Grothendieck topos、Abelian categories、Grothendieck abelian categories、derived categories、presentable categories、局部化、张量范畴和基本拓扑空间。

## 43.1 Profinite 站点与 condensed sets

**定义 43.1.** 记 $\operatorname{ProFin}$ 为 profinite sets 与连续映射组成的范畴。给定有限族连续映射 $\{S_i\to S\}_{i\in I}$，若诱导映射 $\coprod_iS_i\to S$ 为满射，则称其为覆盖族。

**定义 43.2.** Condensed set 是站点 $\operatorname{ProFin}$ 上的 set-valued sheaf。其范畴记为

$$
\operatorname{Cond}(\mathbf{Set})=\operatorname{Shv}(\operatorname{ProFin}).
$$

Condensed abelian group 是 abelian group-valued sheaf，范畴记为 $\operatorname{Cond}(\mathbf{Ab})$。

**外部输入定理 43.3.** 在固定 universe 口径下，$\operatorname{Cond}(\mathbf{Set})$ 是 Grothendieck topos，$\operatorname{Cond}(\mathbf{Ab})$ 是 Grothendieck abelian category。

## 43.2 离散对象的全忠实嵌入

**定义 43.4.** 对集合 $A$，定义 condensed set $\underline A$ 为

$$
\underline A(S)=\operatorname{Map}_{\operatorname{cts}}(S,A_{\operatorname{disc}})
$$

其中 $A_{\operatorname{disc}}$ 赋予离散拓扑。

**命题 43.5.** 函子 $A\mapsto\underline A$ 给出 $\mathbf{Set}$ 到 $\operatorname{Cond}(\mathbf{Set})$ 的全忠实嵌入。

**证明.** 首先，连续映射到离散空间满足 sheaf 条件：若 $\{S_i\to S\}$ 为有限满覆盖，连续映射 $S\to A_{\operatorname{disc}}$ 等价于各 $S_i\to A_{\operatorname{disc}}$ 上连续映射且在 $S_i\times_SS_j$ 上相容，因为连续映射可沿满射覆盖粘合，且离散目标使连续性可在覆盖上检测。

对全忠实性，任意 sheaf 态射 $\eta:\underline A\to\underline B$ 在点 $* \in\operatorname{ProFin}$ 上给出函数 $A=\underline A(*)\to\underline B(*)=B$。反过来，任意函数 $\phi:A\to B$ 与连续映射复合，给出自然变换 $\underline A\to\underline B$。两构造互逆，因此

$$
\operatorname{Hom}_{\operatorname{Cond}}(\underline A,\underline B)\cong\operatorname{Hom}_{\mathbf{Set}}(A,B).
$$

$\square$

## 43.3 拓扑空间与凝聚化函子

**定义 43.6.** 对拓扑空间 $T$，其凝聚化 condensed set 定义为

$$
\underline T(S)=\operatorname{Map}_{\operatorname{cts}}(S,T)
$$

并在满足 sheaf 条件时视为 $\operatorname{Cond}(\mathbf{Set})$ 的对象。

**外部输入定理 43.7.** 对足够好的拓扑空间范畴，例如 compactly generated Hausdorff spaces 的合适子范畴，$T\mapsto\underline T$ 是到 condensed sets 的全忠实嵌入，并与有限极限相容。

**命题 43.8.** 若 $T$ 为离散空间，则定义 43.6 与定义 43.4 一致。

**证明.** 对任意 profinite set $S$，

$$
\underline T(S)=\operatorname{Map}_{\operatorname{cts}}(S,T)
$$

而 $T$ 离散时这正是定义 43.4 中 $\underline A(S)$，其中 $A$ 为 $T$ 的底层集合。$\square$

## 43.4 Condensed abelian groups 的同调代数

**定义 43.9.** 对 condensed set $X$，自由 condensed abelian group $\mathbb Z[X]$ 是 forgetful functor

$$
\operatorname{Cond}(\mathbf{Ab})\to\operatorname{Cond}(\mathbf{Set})
$$

的左伴随作用于 $X$ 的值。

**外部输入定理 43.10.** $\operatorname{Cond}(\mathbf{Ab})$ 有足够投射对象；由 extremally disconnected profinite sets 产生的自由对象给出投射生成族。

**命题 43.11.** $\operatorname{Cond}(\mathbf{Ab})$ 中 filtered colimits exact。

**证明.** 由定理 43.3，$\operatorname{Cond}(\mathbf{Ab})$ 是 Grothendieck abelian category。Grothendieck abelian category 的公理包含 AB5，即 filtered colimits exact。$\square$

## 43.5 Solid modules

**外部输入定理 43.12（Solidification）.** 存在 $\operatorname{Cond}(\mathbf{Ab})$ 的反射性对称幺半局部化

$$
(-)^{\mathrm{solid}}:\operatorname{Cond}(\mathbf{Ab})\to\operatorname{Solid}
$$

其本质像称为 solid abelian groups。该局部化诱导 solid tensor product

$$
M\otimes^{\mathrm{solid}} N=(M\otimes N)^{\mathrm{solid}}.
$$

**定义 43.13.** 设 $A$ 为 solid commutative algebra。Solid $A$-module 是 $\operatorname{Solid}$ 中的 $A$-module。其范畴记作

$$
\operatorname{Mod}^{\mathrm{solid}}_A.
$$

**命题 43.14.** 若 $M,N$ 为 solid abelian groups，则 $M\otimes^{\mathrm{solid}} N$ 仍为 solid。

**证明.** 按定义，$M\otimes^{\mathrm{solid}} N$ 是普通 condensed tensor product $M\otimes N$ 经过 solidification 后的对象。Solidification 的值落在 $\operatorname{Solid}$ 的本质像中，因此 $M\otimes^{\mathrm{solid}} N$ solid。$\square$

## 43.6 Derived solid categories

**定义 43.15.** Solid derived category $D_{\mathrm{solid}}(A)$ 是 solid $A$-modules 的 derived $\infty$-category，或等价地由 condensed $A$-modules 的 derived category 对 solid equivalences 局部化得到的稳定 presentable $\infty$-范畴。

**外部输入定理 43.16.** 对合适 solid algebra $A$，$D_{\mathrm{solid}}(A)$ 是稳定 presentable 对称幺半 $\infty$-范畴，且张量积分别保持小余极限。

**命题 43.17.** Solidification 的导出函子是左伴随，因而保持小余极限。

**证明.** Solidification 是反射性局部化，即包含函子 $i:\operatorname{Solid}\hookrightarrow\operatorname{Cond}(\mathbf{Ab})$ 有左伴随 $(-)^{\mathrm{solid}}$。导出后仍得到左伴随

$$
L(-)^{\mathrm{solid}}:D(\operatorname{Cond}(\mathbf{Ab}))\rightleftarrows D(\operatorname{Solid}):Ri.
$$

任意左伴随保持小余极限，故导出 solidification 保持小余极限。$\square$

## 43.7 解析环与解析范畴

**定义 43.18.** 解析环的范畴论抽象是一个 condensed 或 solid 交换代数 $A$，配有指定的完备化、测试对象和模范畴结构，使 $\operatorname{Mod}^{\mathrm{solid}}_A$ 能同时编码代数和拓扑解析信息。

**外部输入定理 43.19.** Clausen-Scholze 的 analytic rings 形成适合相对解析几何的范畴；其模范畴是 solid/condensed 语境中的稳定 presentable 范畴，并满足良好的基变换和完备性性质。

**命题 43.20.** Condensed 口径把拓扑向量空间问题转化为 sheaf 与模范畴问题。

**证明.** 拓扑对象 $T$ 通过 $S\mapsto\operatorname{Map}_{cts}(S,T)$ 变为 sheaf。若 $T$ 有群、环或向量空间结构，则这些运算逐点给出 condensed group、ring 或 module 结构。于是连续性被吸收到 profinite 测试对象上的函子性和 sheaf 条件中，而同调代数可在 Grothendieck abelian category 或稳定 presentable $\infty$-category 中进行。$\square$

## 43.8 本章小结

Condensed sets 用 profinite 测试对象上的 sheaves 替代点集拓扑。离散集合和许多拓扑空间全忠实嵌入其中，condensed abelian groups 形成 Grothendieck abelian category，solidification 则给出适合解析张量积和完备性问题的反射性局部化。由此，拓扑代数和泛函分析对象可进入 sheaf、module、derived category 和 higher algebra 的统一框架。

## 练习

**练习 43.1.** 定义 $\operatorname{ProFin}$ 站点的覆盖族。

**练习 43.2.** 定义 condensed set 和 condensed abelian group。

**练习 43.3.** 证明离散集合给出 condensed set。

**练习 43.4.** 证明离散嵌入 $\mathbf{Set}\to\operatorname{Cond}(\mathbf{Set})$ 全忠实。

**练习 43.5.** 定义拓扑空间 $T$ 的凝聚化 $\underline T$。

**练习 43.6.** 证明离散拓扑空间的凝聚化与离散集合嵌入一致。

**练习 43.7.** 定义自由 condensed abelian group。

**练习 43.8.** 说明 $\operatorname{Cond}(\mathbf{Ab})$ 为 Grothendieck abelian category 的后果。

**练习 43.9.** 定义 solidification。

**练习 43.10.** 定义 solid tensor product。

**练习 43.11.** 证明 solid tensor product 的值仍为 solid。

**练习 43.12.** 定义 solid $A$-module。

**练习 43.13.** 说明导出 solidification 为什么保持小余极限。

**练习 43.14.** 解释 condensed 口径如何把拓扑向量空间问题转化为 sheaf 问题。
