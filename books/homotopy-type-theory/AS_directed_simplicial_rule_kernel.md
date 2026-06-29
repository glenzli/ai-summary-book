# 附录 AS：Directed / Simplicial Type Theory 的规则核

附录 AN 说明 directed/simplicial type theory 的研究接口。本附录进一步给出对象语言层面的最小规则核，防止把 directed hom 当作普通 identity type 使用。

## AS.1 判断形式

**规则 AS.1（语境与类型判断）.** Directed/simplicial 扩展保留普通 HoTT 的判断：
$$
\Gamma\ \mathsf{ctx},\qquad
\Gamma\vdash A:\mathcal U,\qquad
\Gamma\vdash a:A.
$$
此外引入 directed hom 类型形成规则。

**规则 AS.2（directed hom 形成）.** 若
$$
\Gamma\vdash A:\mathcal U,\qquad
\Gamma\vdash a:A,\qquad
\Gamma\vdash b:A,
$$
则
$$
\Gamma\vdash \mathsf{hom}_A(a,b):\mathcal U.
$$

**规则 AS.3（identity 与 hom 的分离）.** 普通恒等类型
$$
a=_A b
$$
和 directed hom
$$
\mathsf{hom}_A(a,b)
$$
是不同类型形成子。除非额外给定离散性或 core 结构，不存在一般映射
$$
\mathsf{hom}_A(a,b)\to(a=b)
$$
或
$$
\mathsf{hom}_A(a,b)\to\mathsf{hom}_A(b,a).
$$

## AS.2 恒等态射与组合

**规则 AS.4（directed identity）.** 对 $a:A$，有
$$
\mathsf{id}^d_a:\mathsf{hom}_A(a,a).
$$

**输入 AS.5（composition）.** 对 $a,b,c:A$，有组合
$$
\circ^d:
\mathsf{hom}_A(b,c)\to
\mathsf{hom}_A(a,b)\to
\mathsf{hom}_A(a,c).
$$

**输入 AS.6（单位与结合相干）.** 组合满足左右单位律和结合律，但这些通常不是 judgmental equality，而是由 simplicial horn filler 给出的高阶相干：
$$
u\circ^d\mathsf{id}^d_a\simeq u,\qquad
\mathsf{id}^d_b\circ^d u\simeq u,
$$
$$
(w\circ^d v)\circ^d u\simeq w\circ^d(v\circ^d u).
$$

**边界.** 在 Segal type 口径中，组合和相干可以由 unique horn fillers 派生，而不是作为原始构造子全部加入。

## AS.3 Segal 条件

**定义 AS.7（spine map 口径）.** 对 $n\ge2$，设 $\Delta^n$ 为形式 $n$-单纯形，$I^n\subseteq\Delta^n$ 为 spine。类型 $A$ 满足 Segal 条件，若 restriction
$$
A^{\Delta^n}\to A^{I^n}
$$
在合适 directed exponent 口径下为等价。

**等价形式 AS.8（唯一组合）.** 对 $n=2$，Segal 条件给出：任意 composable pair
$$
u:\mathsf{hom}_A(a,b),\qquad
v:\mathsf{hom}_A(b,c)
$$
有可收缩的 composite filler 类型。其中心给出 $v\circ^d u$。

**定理 AS.9（高阶相干由低阶 horn filler 推出，外部输入）.** 在 de Jong-Kraus-Ljungström 的 HoTT+interval 口径下，唯一 $(2,1)$-horn fillers 推出所有 inner $(n,k)$-horn fillers 的唯一性。

**使用边界.** 这是附录 AN.4 的规则级版本。它允许把 AS.5-AS.6 看成从 horn filler 原理派生，但不允许在普通 HoTT 中无条件使用 directed composition。

## AS.4 Directed universe

**输入 AS.10（离散类型宇宙）.** Simplicial/directed 口径中有离散类型宇宙
$$
\mathcal S:\mathcal U
$$
其对象为 ordinary types 的某个嵌入或离散化。

**输入 AS.11（directed univalence）.** 对 $A,B:\mathcal S$，有等价
$$
\mathsf{hom}_{\mathcal S}(A,B)\simeq(A\to B).
$$

**对比 AS.12（普通 univalence）.** 普通 HoTT 的单值性是
$$
(A=B)\simeq(A\simeq B).
$$
Directed univalence 是
$$
\mathsf{hom}_{\mathcal S}(A,B)\simeq(A\to B).
$$
二者目标不同：前者把 identity path 识别为 equivalence，后者把 directed hom 识别为 function。

## AS.5 Displayed directed structures

**定义 AS.13（displayed directed type）.** 在 directed base $A$ 上的 displayed directed type 由族
$$
P:A\to\mathcal U
$$
和对每个 directed arrow $u:\mathsf{hom}_A(a,b)$ 的 transport-like 函数
$$
u_\ast:P(a)\to P(b)
$$
组成，并满足 identity 和 composition 相干。

**命题 AS.14（结构同态原则的规则形态）.** 若结构 $P$ 是 displayed directed type，则 directed arrow $u:\mathsf{hom}_A(a,b)$ 自动诱导结构 transport
$$
P(a)\to P(b).
$$

**证明.** 这是定义 AS.13 中 $u_\ast$ 的应用。identity 与 composition 相容性由 displayed directed type 的相干给出。$\square$

## AS.6 与本书普通 HoTT 的接口

1.  第 1-14 章默认不包含 AS.2-AS.11。
2.  若只需普通范畴论，使用第十三章的单值范畴，不使用 directed hom。
3.  若需要 $\infty$-范畴对象语言，必须在章节开头声明切换到 directed/simplicial 口径，并列出 AS.1-AS.11 中采用的规则。
4.  任何从 directed hom 推出 identity path 的步骤都必须给出额外离散性、core 或 equivalence 数据。

