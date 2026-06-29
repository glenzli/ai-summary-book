# 第十七章：Dendroidal inner Kan 条件与 homotopy operads

定义 16.14--命题 16.18 把 strict colored operad 嵌入 dendroidal sets。本章放松 strict composition：不要求每棵树的全局运算由顶点运算严格唯一决定，而只要求缺少一个 inner face 的 horn 能被填充。这个条件是 quasi-category inner horn 条件的 operadic analogue。

## 17.1 Inner horn fillers

**定义 17.1.** 设 $X$ 是 dendroidal set，$T$ 是树，$e$ 是 $T$ 的 inner edge。一个 $e$-inner horn in $X$ 是一个态射
$$
\Lambda^e[T]\to X.
$$
一个 filler 是使图表交换的延拓
$$
\begin{CD}
\Lambda^e[T] @>>> X\\
@VVV @.\\
\Omega[T] @>>> X.
\end{CD}
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
有 filler。在线性树嵌入 $i:\Delta\to\Omega$ 下，$\Lambda^k[n]$ 对应线性树 $L_n$ 中第 $k$ 个 inner edge 的 dendroidal inner horn。给出 $\Lambda^k[n]\to i^\*X$ 等价于给出 $\Lambda^e[L_n]\to X$。由于 $X$ inner Kan，该 dendroidal horn 有 filler $\Omega[L_n]\to X$。限制回 $\Delta$ 即得到 simplicial filler $\Delta[n]\to i^\*X$。$\square$

## 17.2 Strict operads give unique fillers

**定理 17.5.** 对任意 colored operad $\mathcal P$，其 dendroidal nerve $N_d(\mathcal P)$ 是 inner Kan，并且每个 inner horn 有唯一 filler。Moerdijk--Weiss 定位为 MW-4。

**证明.** 设给定 horn
$$
\Lambda^e[T]\to N_d(\mathcal P).
$$
通过 Yoneda，这等价于为除 inner face $\partial_e:T/e\to T$ 之外的所有 elementary faces 指定 compatible operations in $\mathcal P$。

由命题 16.18 的严格 Segal 性，给出 $\Omega[T]\to N_d(\mathcal P)$ 等价于：

1. 为每条边 $a\in E(T)$ 指定颜色 $c_a$；
2. 为每个顶点 $v\in V(T)$ 指定运算
   $$
   p_v\in\mathcal P((c_a)_{a\in\operatorname{in}(v)};c_{\operatorname{out}(v)}).
   $$

Horn 中所有 outer faces 和除 $\partial_e$ 外的 inner faces 已经包含每个顶点 corolla 的信息，也包含共享边上的颜色一致性信息。因此 horn 数据确定全部 $c_a$ 与 $p_v$。由这些数据存在唯一 operad morphism
$$
\Omega(T)\to\mathcal P
$$
并给出 filler $\Omega[T]\to N_d(\mathcal P)$。

唯一性同样来自严格 Segal 性：任一 filler 的限制到顶点 corollas 必须等于 horn 已给数据，而这些 corolla 数据唯一决定 $\Omega(T)\to\mathcal P$。$\square$

**说明 17.6.** 定理 17.5 是“strict operad 是 infinity-operad 的特殊情形”的精确表达。唯一 filler 反映 strict composition；一般 inner Kan dendroidal set 只要求 filler 存在，组合不再严格唯一。

## 17.3 Homotopy coherent operations

设 $X$ 是 inner Kan dendroidal set。颜色集合定义为
$$
\operatorname{Col}(X)=X_\eta.
$$
对颜色 $c_1,\ldots,c_n,c$，定义 operations space 的集合层级近似为 fiber
$$
X(c_1,\ldots,c_n;c)
=
\{x\in X_{C_n}: \partial_i x=c_i,\ \partial_{\mathrm{out}}x=c\}.
$$
这里 $\partial_i:\eta\to C_n$ 表示第 $i$ 条输入边的颜色，$\partial_{\mathrm{out}}:\eta\to C_n$ 表示输出边的颜色。

**说明 17.7.** 上式只是集合层级的 fiber。若要得到真正的 mapping spaces，需要在 dendroidal set 的 simplicial enrichment、slice 或 derived mapping object 中构造同伦 fiber。本书后续只在需要时引入该技术。

**命题 17.8.** Inner horn filler 给出 operations 的同伦相干复合。

**证明.** 考虑一棵有两个顶点并由 inner edge $e$ 相连的树 $T$。给定两个可复合 operations，即给定两个顶点 corollas 到 $X$ 的元素，并要求共享 inner edge 的颜色一致。这些数据给出 horn
$$
\Lambda^e[T]\to X
$$
的一部分；加上外面 faces 的兼容数据后得到完整 inner horn。Inner Kan 条件给出 filler
$$
\Omega[T]\to X.
$$
该 filler 在 inner face $T/e$ 上的限制是一个单顶点 corolla 的元素，表示两个 operations 的复合。不同 filler 可能给出不同复合，但它们被更高维树上的 horn fillers 组织成同伦相干关系。$\square$

**警告 17.9.** Inner Kan 条件不说复合唯一，也不说所有 choices 形成可缩空间。若需要唯一 up to contractible choice，需要更强的 fibrancy、mapping space 或 completeness 条件，并必须说明所在模型。

## 17.4 Normal monomorphisms

**定义 17.10.** 对树 $T$，其 automorphism group $\operatorname{Aut}_\Omega(T)$ 作用在 $X_T$ 上。一个元素 $x\in X_T$ 称为 degenerate，若它来自某个非同构 degeneracy map。否则称为 nondegenerate。

**定义 17.11.** Dendroidal set $X$ 称为 normal，若对每个树 $T$，$\operatorname{Aut}_\Omega(T)$ 在 nondegenerate elements of $X_T$ 上自由作用。

Monomorphism $A\to X$ 称为 normal monomorphism，若对每个树 $T$，$\operatorname{Aut}_\Omega(T)$ 在 $X_T\setminus A_T$ 的 nondegenerate classes 上自由作用。

**说明 17.12.** Simplicial sets 中 cofibrations 是 monomorphisms。Dendroidal sets 中必须用 normal monomorphisms 替代所有 monomorphisms，原因是树可能有非平凡 automorphisms，对称群稳定子会破坏等变胞腔论证。

**外部输入命题 17.13.** 对每棵树 $T$ 和 inner edge $e$，horn inclusion
$$
\Lambda^e[T]\hookrightarrow\Omega[T]
$$
是 normal monomorphism。

**证明边界.** 它逐树为 representable subpresheaf 的包含，因此是 monomorphism。正规性还需要检查所有未落入 horn 的 nondegenerate dendrexes 上没有非平凡树自同构稳定子；这属于 $\Omega$ 的 elementary face 与 degeneracy 分解理论。完整证明依赖 Moerdijk-Weiss/Cisinski-Moerdijk 的 normal monomorphism 引理，本书将其作为外部输入。$\square$

## 17.5 Dendroidal operadic model structure

**定义 17.14.** Inner anodyne maps 是由所有 inner horn inclusions
$$
\Lambda^e[T]\hookrightarrow\Omega[T]
$$
通过 pushout、transfinite composition 和 retract 生成的态射类。

**外部输入定理 17.15（Cisinski-Moerdijk）.** $\mathbf{dSet}$ 上存在 operadic model structure，使得：

1. cofibrations 是 normal monomorphisms；
2. fibrant objects 是 inner Kan dendroidal sets；
3. inner horn inclusions 是 trivial cofibrations；
4. weak equivalences 称为 operadic weak equivalences。

该模型结构是 left proper、cofibrantly generated，并给出 homotopy operads 的模型。

**说明 17.16.** Operadic weak equivalence 不能简单定义为逐树双射或逐树弱等价，因为 dendroidal set 是集合值对象。其定义通过模型结构、local objects 或 derived mapping spaces 给出。本书在使用该词时总指 Cisinski-Moerdijk operadic model structure 中的 weak equivalence。

**命题 17.17.** 若 $A\to X$ 是 inner anodyne map 且 $Y$ 是 inner Kan dendroidal set，则任意交换方块
$$
\begin{CD}
A @>>> Y\\
@VVV @.\\
X @>>> *
\end{CD}
$$
有 lift $X\to Y$。

**证明.** Inner anodyne maps 由 inner horn inclusions 经 pushout、transfinite composition 和 retract 生成。对生成元 $\Lambda^e[T]\to\Omega[T]$，lift 的存在正是 $Y$ 的 inner Kan 条件。具有左提升性质的态射类对 pushout、transfinite composition 和 retract 稳定。因此所有 inner anodyne maps 对 $Y\to *$ 有左提升性质。$\square$

## 17.6 Strict operads inside the model structure

**外部输入定理 17.18（模型比较入口）.** 集合值 strict operads 通过 dendroidal nerve
$$
N_d:\operatorname{Operad}_{\mathrm{col}}\to\mathbf{dSet}
$$
嵌入 $\mathbf{dSet}$，fully faithfulness 定位为 MW-2。对 simplicial operads、topological operads 或 Lurie-style infinity-operads，需要使用 homotopy coherent dendroidal nerve 或相应比较构造；homotopy coherent dendroidal nerve 的 inner Kan 入口定位为 MW-5，dendroidal--Lurie 比较按 P0 引用定位批次 10 中 HHM-1--HHM-5 使用，并需保留模型结构与 fibrancy 假设。

**说明 17.19.** 对集合值 strict operads，$N_d(\mathcal P)$ 是具有唯一 inner fillers 的 fibrant object。对 simplicial operads，不能把每个 simplicial operation space 只取 $0$-simplices 后当作完整同伦理论；这样会丢失高阶 simplex 所编码的同伦信息。

**命题 17.20.** 若 $\mathcal P$ 是 ordinary category 视为 unary colored operad，则 $N_d(\mathcal P)$ 的线性限制是 ordinary nerve，并且定理 17.5 限制为 ordinary nerve 中 inner horns 的唯一填充性质。

**证明.** 第一部分是命题 16.28。若 ordinary category 的 nerve 中给定 inner horn，则缺失的 face 对应一段可复合箭头的复合。范畴中复合存在且唯一，并满足结合律，因此该 horn 有唯一 filler。这个唯一 filler 与定理 17.5 中由 unary operad 复合给出的 filler 相同。$\square$

**警告 17.21.** Quasi-category 的 nerve 若来自 ordinary category，则 inner horn fillers 唯一；一般 quasi-category 只要求 filler 存在。Dendroidal 情形完全类似：strict operad 给唯一 fillers，infinity-operad 只给存在性。

## 17.7 本章小结

Dendroidal inner Kan 条件把 operad 的复合从严格唯一改为 horn filler 的存在。Strict operads 的 dendroidal nerve 有唯一 inner fillers；一般 dendroidal infinity-operad 的 fillers 表示 homotopy coherent composition。Cisinski-Moerdijk 模型结构把这些对象组织成 homotopy operads 的模型范畴，其中 cofibrations 是 normal monomorphisms，fibrant objects 是 inner Kan dendroidal sets。

## 练习

**练习 17.1.** 对两顶点树 $T$，写出 inner horn $\Lambda^e[T]$ 包含哪些 elementary faces。

**练习 17.2.** 证明 ordinary category nerve 的 inner horn fillers 唯一。

**练习 17.3.** 给出一个有非平凡 automorphism 的 corolla，并解释 normality 条件为何涉及自由作用。

**练习 17.4.** 证明 inner anodyne maps 对 inner Kan objects 有左提升性质。

**练习 17.5.** 解释为什么 inner Kan 条件不等价于 strict operad nerve 条件。
