# 第十七章：Dendroidal inner Kan 条件与 homotopy operads

对一棵两顶点树，strict operad 会把两个顶点运算唯一复合成收缩内边后的运算；一般同伦对象中，这个复合不应预先选成严格唯一的元素。Dendroidal inner horn 正好保留除某个内面外的全部兼容数据，而 filler 补回该面。要求所有 inner horns 可填，便把“存在相干复合”与“复合严格唯一”分离开来。本章由两顶点计算进入 inner Kan 条件，再说明 normal monomorphism 为什么控制树自同构，并以 Cisinski--Moerdijk 模型结构把 raw filler 集合提升到可谈派生映射空间的同伦理论。线性树限制将同时恢复 quasi-category 的 inner horn 图景。

## 17.1 Inner horn fillers

**定义 17.1.** 设 $X$ 是 dendroidal set，$T$ 是树，$e$ 是 $T$ 的 inner edge。一个 $e$-inner horn in $X$ 是一个态射
$$
\Lambda^e[T]\to X.
$$
一个 filler 是使图表交换的延拓
$$
\begin{array}{ccc}
\Lambda^e[T] & \longrightarrow & X\\
\downarrow & & \Vert\\
\Omega[T] & \longrightarrow & X
\end{array}
$$

**定义 17.2.** Dendroidal set $X$ 称为 inner Kan，或称为 dendroidal infinity-operad，若对每个树 $T$ 和每条 inner edge $e$，任意 horn
$$
\Lambda^e[T]\to X
$$
都存在 filler。

**说明 17.3.** 本书用“dendroidal infinity-operad”表示 inner Kan dendroidal set。Lurie-style infinity-operad 是另一种模型；二者的比较是外部输入定理 18.20，而不是定义相同。

**命题 17.4.** 若 $X$ 是 inner Kan dendroidal set，则其线性限制
$$
i^\*X\in\mathbf{sSet}
$$
是 quasi-category。

**证明.** Quasi-category 条件要求每个 simplicial inner horn
$$
\Lambda^k[n]\to i^\*X,\qquad 0<k<n,
$$
有 filler。由 MW-1 的全忠实线性树嵌入和 MW-3 的 extension-by-zero 识别，
$$
i_!\Delta[n]=\Omega[L_n],
\qquad
i_!\Lambda^k[n]=\Lambda^e[L_n],
$$
其中 $e$ 是 $L_n$ 的第 $k$ 条 inner edge。伴随给出
$$
\mathbf{sSet}(\Lambda^k[n],i^*X)
\cong
\mathbf{dSet}(\Lambda^e[L_n],X).
$$
由于 $X$ inner Kan，右侧 horn map 延拓到 $\Omega[L_n]\to X$；再由伴随限制回 $\Delta[n]\to i^*X$。$\square$

## 17.2 Strict operads give unique fillers

**外部输入定理 17.5（strict nerve 的唯一 inner fillers；MW-4）.** 对任意集合值 colored operad $\mathcal P$，其 dendroidal nerve $N_d(\mathcal P)$ 是 inner Kan，并且每个 inner horn 有唯一 filler。来源为 Moerdijk--Weiss, arXiv:math/0701293v2, Example 7.1（定位 MW-4）。

**证明路线（外部输入）.** 来源用 free colored operad $\Omega(T)$ 的生成元和 face 关系证明：horn 的兼容面数据唯一确定缺失 inner edge 上的 strict composite。命题 T.1--命题 T.2 在两顶点树上完成该重建；一般树还需 elementary face/degeneracy 分解，本文不把两顶点论证冒充为全树证明。

**说明 17.6.** 定理 17.5 是“strict operad 是 infinity-operad 的特殊情形”的精确表达。唯一 filler 反映 strict composition；一般 inner Kan dendroidal set 只要求 filler 存在，组合不再严格唯一。

## 17.3 Homotopy coherent operations

设 $X$ 是 inner Kan dendroidal set。颜色集合定义为
$$
\operatorname{Col}(X)=X_\eta.
$$
对颜色 $c_1,\ldots,c_n,c$，定义它们之间的**运算集**（operation set）为普通集合纤维
$$
\operatorname{Op}_X(c_1,\ldots,c_n;c)
=
X_{C_n}\times_{X_\eta^{n+1}}
\{(c_1,\ldots,c_n,c)\}.
$$
这里 $X_{C_n}\to X_\eta^{n+1}$ 由 $n$ 条输入边和一条输出边的树映射
$$
\partial_i:\eta\to C_n,
\qquad
\partial_{\mathrm{out}}:\eta\to C_n
$$
诱导。换言之，$\operatorname{Op}_X(c_1,\ldots,c_n;c)$ 的元素只是具有指定边颜色的 corolla dendrex；此定义没有给它附加拓扑或 simplicial 方向。

**说明 17.7（派生运算空间）.** 记
$$
\iota_E:
\coprod_{e\in E(C_n)}\Omega[\eta]\longrightarrow\Omega[C_n]
$$
为所有边包含诱导的 dendroidal set 态射。固定颜色元组
$$
\mathbf c=(c_1,\ldots,c_n,c):
\coprod_{e\in E(C_n)}\Omega[\eta]\longrightarrow X.
$$
在外部输入定理 17.15 的 operadic model structure 中选取任一同伦函数复形
$\operatorname{Map}^{\mathbf R}_{\mathbf{dSet}}(-,-)$，定义派生运算空间
$$
\operatorname{Op}^{h}_X(c_1,\ldots,c_n;c)
:=
\operatorname{hofib}_{\mathbf c}\!\left(
\operatorname{Map}^{\mathbf R}_{\mathbf{dSet}}(\Omega[C_n],X)
\longrightarrow
\operatorname{Map}^{\mathbf R}_{\mathbf{dSet}}
\left(\coprod_{e\in E(C_n)}\Omega[\eta],X\right)
\right).
$$
同伦函数复形的不同构造给出弱等价的空间，故上式的弱同伦型良定义。普通纤维 $\operatorname{Op}_X$ 与这个同伦纤维不是同一个定义；后文若只使用集合层信息，一律写 operation set。

**命题 17.8（两顶点 horn 的复合集）.** 设 $T$ 是由两个顶点沿 inner edge $e$ 连接的树，$X$ 是 inner Kan dendroidal set，并且
$$
a:\Lambda^e[T]\to X
$$
是一个完整 horn map。令 $\operatorname{Fill}(a)$ 为其 filler 集。每个 $\bar a\in\operatorname{Fill}(a)$ 沿缺失 inner face 限制为
$$
(\partial_e)^*\bar a\in X_{T/e}.
$$
因此复合集
$$
\operatorname{Comp}(a)
=
\{(\partial_e)^*\bar a:\bar a\in\operatorname{Fill}(a)\}
\subseteq X_{T/e}
$$
非空。若 $X=N_d(\mathcal P)$，则由外部输入定理 17.5，$\operatorname{Fill}(a)$ 与 $\operatorname{Comp}(a)$ 都是单点，后者是 $\mathcal P$ 中的 strict composite。

**证明.** Inner Kan 条件直接给出 $\operatorname{Fill}(a)\ne\varnothing$。对每个 filler 预合成 face map $\Omega[T/e]\to\Omega[T]$，得到显示的 $X_{T/e}$ 元素。Strict nerve 情形由唯一 filler 和命题 T.2 的两顶点计算给出。$\square$

**警告 17.9.** 两个颜色相容的 corolla 元素首先只给出 Segal core map
$$
\operatorname{Sc}[T]\to X,
$$
不自动给出完整 horn map $\Lambda^e[T]\to X$；要应用命题 17.8，必须给出其余 outer-face 兼容数据，或引用能把 Segal core extension 成 horn 的定理。Inner Kan 定义只断言 raw filler set 非空，不断言严格唯一。若在 operadic model structure 的 simplicial derived mapping spaces 中讨论 fillers，则 horn inclusion 为 trivial cofibration、$X$ fibrant 可给出 derived filler space 的可缩性；这是模型范畴结论，不是 raw 集合层定义。

## 17.4 Normal monomorphisms

**定义 17.10.** 对树 $T$，其 automorphism group $\operatorname{Aut}_\Omega(T)$ 作用在 $X_T$ 上。一个元素 $x\in X_T$ 称为 degenerate，若它来自某个非同构 degeneracy map。否则称为 nondegenerate。

**定义 17.11.** Dendroidal set $X$ 称为 normal，若对每个树 $T$，$\operatorname{Aut}_\Omega(T)$ 在 nondegenerate elements of $X_T$ 上自由作用。

Monomorphism $A\to X$ 称为 normal monomorphism，若对每个树 $T$，$\operatorname{Aut}_\Omega(T)$ 在 $X_T\setminus A_T$ 的 nondegenerate classes 上自由作用。

**说明 17.12.** Simplicial sets 中 cofibrations 是 monomorphisms。Dendroidal sets 中必须用 normal monomorphisms 替代所有 monomorphisms，原因是树可能有非平凡 automorphisms，对称群稳定子会破坏等变胞腔论证。

**外部输入命题 17.13（inner horn 的 normality；CM-1--CM-2）.** 对每棵树 $T$ 和 inner edge $e$，horn inclusion
$$
\Lambda^e[T]\hookrightarrow\Omega[T]
$$
是 normal monomorphism。

**证明路线（外部输入）.** 它逐树为 representable subpresheaf 的包含，故 monomorphism 部分可直接检查。Normality 还需分析所有未落入 horn 的 nondegenerate dendrexes 的树自同构稳定子；该步骤使用 $\Omega$ 的 elementary face/degeneracy 分解和 CM-1--CM-2，本书不重证。

## 17.5 Dendroidal operadic model structure

**定义 17.14.** Inner anodyne maps 是由所有 inner horn inclusions
$$
\Lambda^e[T]\hookrightarrow\Omega[T]
$$
通过 pushout、transfinite composition 和 retract 生成的态射类。

**外部输入定理 17.15（Cisinski--Moerdijk operadic model structure；CM-3）.** $\mathbf{dSet}$ 上存在 operadic model structure，使得：

1. cofibrations 是 normal monomorphisms；
2. fibrant objects 是 inner Kan dendroidal sets；
3. inner horn inclusions 是 trivial cofibrations；
4. weak equivalences 称为 operadic weak equivalences。

本书使用的存在性与 fibrant/cofibrant 描述定位为 Cisinski--Moerdijk, arXiv:0902.1954v2, Theorem 2.4（CM-3）。涉及 left properness、monoidal compatibility 或 fibrant objects 间的弱等价判别时，必须分别引用来源相应命题及 erratum；这些附加性质不从上列四项形式推出。

**说明 17.16.** Operadic weak equivalence 不能简单定义为逐树双射或逐树弱等价，因为 dendroidal set 是集合值对象。其定义通过模型结构、local objects 或 derived mapping spaces 给出。本书在使用该词时总指 Cisinski-Moerdijk operadic model structure 中的 weak equivalence。

**命题 17.17.** 若 $A\to X$ 是 inner anodyne map 且 $Y$ 是 inner Kan dendroidal set，则任意交换方块
$$
\begin{array}{ccc}
A & \longrightarrow & Y\\
\downarrow & & \downarrow\\
X & \longrightarrow & *
\end{array}
$$
有 lift $X\to Y$。

**证明.** Inner anodyne maps 由 inner horn inclusions 经 pushout、transfinite composition 和 retract 生成。对生成元 $\Lambda^e[T]\to\Omega[T]$，lift 的存在正是 $Y$ 的 inner Kan 条件。具有左提升性质的态射类对 pushout、transfinite composition 和 retract 稳定。因此所有 inner anodyne maps 对 $Y\to *$ 有左提升性质。$\square$

## 17.6 Strict operads inside the model structure

**外部输入定理 17.18（strict 与 simplicial nerve 入口；MW-2、MW-5）.** 集合值 strict operads 通过 dendroidal nerve
$$
N_d:\operatorname{Operad}_{\mathrm{col}}\to\mathbf{dSet}
$$
全忠实嵌入 $\mathbf{dSet}$；来源为 MW-2。若 $\mathcal P$ 是 fibrant simplicial operad，则其 homotopy coherent dendroidal nerve 为 inner Kan 的入口定位为 MW-5。两条结论都不是 dendroidal--Lurie 模型比较。

**外部输入定理 17.18.1（dendroidal--Lurie 比较边界；HHM-1--HHM-5）.** Heuts--Hinich--Moerdijk 给出经过 simplicial operads、forest sets、marked open forest sets 与 preoperads 的 Quillen-equivalence zig-zag；将该 zig-zag解释为 dendroidal--Lurie 比较时，必须保留来源的 open/no-constants 限制。因为本书默认允许 arity $0$，不得把 17.18.1 直接应用于全书默认 operad；必须先限制到 open 子理论，或另引覆盖 nullary operations 的比较定理。

**说明 17.19.** 对集合值 strict operads，$N_d(\mathcal P)$ 是具有唯一 inner fillers 的 fibrant object。对 simplicial operads，不能把每个 simplicial operation space 只取 $0$-simplices 后当作完整同伦理论；这样会丢失高阶 simplex 所编码的同伦信息。

**命题 17.20.** 若 $\mathcal P$ 是 ordinary category 视为 unary colored operad，则 $N_d(\mathcal P)$ 的线性限制是 ordinary nerve，并且定理 17.5 限制为 ordinary nerve 中 inner horns 的唯一填充性质。

**证明.** 第一部分是命题 16.28。若 ordinary category 的 nerve 中给定 inner horn，则缺失的 face 对应一段可复合箭头的复合。范畴中复合存在且唯一，并满足结合律，因此该 horn 有唯一 filler。这个唯一 filler 与定理 17.5 中由 unary operad 复合给出的 filler 相同。$\square$

**警告 17.21.** Quasi-category 的 nerve 若来自 ordinary category，则 inner horn fillers 唯一；一般 quasi-category 只要求 filler 存在。Dendroidal 情形完全类似：strict operad 给唯一 fillers，infinity-operad 只给存在性。

## 17.7 从 filler 集到派生运算空间

两顶点 inner horn 的 filler 给出可能的复合集，但 inner Kan 条件只保证这个集合非空；它既不保证唯一，也不单凭集合纤维产生 operation space。Strict nerve 的唯一 filler 恰好恢复普通 operad 的严格代入。要表达一般 infinity-operad 中选择之间的高阶同伦，必须进入 operadic model structure，并取说明 17.7 的派生映射空间同伦纤维。Normal monomorphism 控制树自同构造成的稳定子，inner horns 成为 trivial cofibrations，inner Kan dendroidal sets 则成为 fibrant objects。下一章将用完全不同的基范畴 $\mathbf{Fin}_*$ 编码同一类多输入现象，模型之间的移动只能通过明示的比较定理完成。

## 练习

**练习 17.1.** 对两顶点树 $T$，写出 inner horn $\Lambda^e[T]$ 包含哪些 elementary faces。

**练习 17.2.** 证明 ordinary category nerve 的 inner horn fillers 唯一。

**练习 17.3.** 给出一个有非平凡 automorphism 的 corolla，并解释 normality 条件为何涉及自由作用。

**练习 17.4.** 证明 inner anodyne maps 对 inner Kan objects 有左提升性质。

**练习 17.5.** 解释为什么 inner Kan 条件不等价于 strict operad nerve 条件。
