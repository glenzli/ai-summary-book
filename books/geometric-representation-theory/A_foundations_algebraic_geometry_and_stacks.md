# 附录 A：代数几何、导出范畴和商栈基础约定

## 本章目标

本附录固定本书关于代数簇、商栈、equivariant sheaves、导出范畴和六函子的基础语言。这里的目标不是重建代数几何，而是确保正文中的每个几何范畴都有明确类型。

## 依赖前置知识

需要熟悉 scheme、morphism、fiber product、abelian category、derived category 和基本 sheaf language。

## A.1 代数簇和群作用

**约定 A.1.** 本书中的代数簇指 $k$ 上分离、有限型、约化 scheme。若需要非约化 scheme、ind-scheme 或 derived stack，会在该处显式说明。

**定义 A.2.** 令 $H$ 为代数群，$X$ 为 $k$-代数簇。一个左 $H$-作用是 morphism
$$
a:H\times X\to X
$$
满足
$$
a(e,x)=x,\qquad a(h_1,a(h_2,x))=a(h_1h_2,x)
$$
作为 $H\times H\times X$ 上的 morphism 恒等式。

**定义 A.3.** 若 $x\in X(k)$，其稳定子为 functor
$$
\operatorname{Stab}_H(x)(R)=\{h\in H(R)\mid h\cdot x_R=x_R\}.
$$
当该 functor 由 closed subgroup scheme 表示时，仍记为 $\operatorname{Stab}_H(x)$。

**引理 A.4.** 若 $H$ 是代数群且 $K\subset H$ 是 closed subgroup，则 $H$ 左乘作用在齐性空间 $H/K$ 上的基点 $eK$ 的稳定子为 $K$。

**证明.** 对任意 $k$-代数 $R$，元素 $h\in H(R)$ 固定 $eK\in(H/K)(R)$ 当且仅当 $hK_R=K_R$。这等价于 $h\in K(R)$。因此稳定子 functor 与 $K$ 的点 functor 相同，故由 Yoneda 引理得到稳定子等于 $K$。$\square$

## A.2 商栈和 equivariant 对象

**定义 A.5.** 令 $H$ 左作用于 $X$。商栈 $[X/H]$ 是如下 fibered category：对任意测试 scheme $S$，$[X/H](S)$ 的对象为一个 $H$-torsor $P\to S$ 和一个 $H$-equivariant morphism $P\to X$。态射为 torsor 同构并与到 $X$ 的映射相容。

**例 A.6.** 若 $X=\operatorname{Spec}k$，则 $[\operatorname{Spec}k/H]=BH$ 是 classifying stack。$BH(S)$ 的对象是 $S$ 上的 $H$-torsors。

**定义 A.7.** 本书把 $H$-equivariant constructible derived category 记为
$$
D^b_H(X,E):=D^b_c([X/H],E),
$$
其中右侧按章节采用 Betti、etale 或其他 sheaf theory 模型。若 $H$ 的作用和 sheaf theory 不支持这个简写，会在正文中使用 descent datum 明确定义。

**命题 A.8.** 若 $X=H/K$ 且 $K$ 为 closed subgroup，则 quotient stack $[X/H]$ 等价于 $BK$。

**证明.** 由 $X=H/K$，给出 $S$ 上的对象 $(P\to S,\phi:P\to H/K)$ 等价于给出 $P$ 的 $K$-reduction。具体地，令
$$
Q=P\times_{H/K} H,
$$
其中 $H\to H/K$ 是右 $K$-torsor。因为 $\phi$ 是 $H$-equivariant，$Q$ 自然是 $S$ 上的 $K$-torsor。反向地，给定 $K$-torsor $Q$，诱导 $H$-torsor $P=Q\times^K H$，并有自然 $H$-equivariant map $P\to H/K$。这两个构造在态射上互逆，故得到 stack 等价。$\square$

**推论 A.9.** $H$-equivariant local systems on $H/K$ 等价于 $K$ 的有限维表示，前提是采用的 sheaf theory 中 $BK$ 上的 local systems 与 $K$-表示相符。

**证明.** 由命题 A.8，$D_H(H/K)$ 可视为 $D(BK)$。local systems 子范畴在该等价下对应 $BK$ 上 local systems。最后一步是所选 sheaf theory 的标准识别，若 $K$ 非离散或系数不合适，则需额外假设。$\square$

## A.3 导出范畴和六函子

**约定 A.10.** 对 morphism $f:X\to Y$，六函子写作
$$
f^\ast,\quad f_\ast,\quad f_!,\quad f^!,\quad -\otimes-,\quad \mathcal Hom.
$$
这些 functor 在正文中总是指相应 sheaf theory 的导出函子。

**外部输入定理 A.11.** 在通常的 constructible sheaf theory 假设下，六函子满足 adjunction、projection formula、proper base change、smooth base change 和 Verdier duality。  
本书不会在基础章节重证该定理；后续附录 C 需要给出具体来源和假设表。

**定义 A.12.** Verdier duality functor 记为
$$
\mathbb D_X(-)=\mathcal Hom(-,\omega_X^\bullet),
$$
其中 $\omega_X^\bullet$ 为 dualizing complex。若 $X$ 光滑纯维数 $d$ 且系数为域，则在常见 convention 下
$$
\omega_X^\bullet\simeq E_X[2d]
$$
用于 complex topology；在 etale 语境中还需 Tate twist，正文会显式标出。

## A.4 卷积 correspondence 的模板

**定义 A.13.** 令
$$
X\xleftarrow{p} Z\xrightarrow{q} Y
$$
为 correspondence。若 $p,q$ 处在支持六函子的语境中，则它定义 functor
$$
\Phi_Z:D(X)\to D(Y),\qquad \Phi_Z(\mathcal F)=q_!p^\ast\mathcal F.
$$
若需要 extraordinary pullback 或 $q_\ast$，必须在正文中说明原因。

**定义 A.14.** 两个 correspondence
$$
X\xleftarrow{p} Z\xrightarrow{q} Y,\qquad
Y\xleftarrow{r} Z'\xrightarrow{s} U
$$
的复合为 fiber product correspondence
$$
X\xleftarrow{p\circ\operatorname{pr}_1} Z\times_Y Z'
\xrightarrow{s\circ\operatorname{pr}_2} U.
$$

**命题 A.15.** 若 proper base change 和 projection formula 在当前 sheaf theory 中成立，且所有 fiber product 均在允许的几何范畴中，则 correspondence 的复合给出的 functor 与 functor 复合自然同构。

**证明.** 对 $\mathcal F\in D(X)$，
$$
\Phi_{Z'}(\Phi_Z(\mathcal F))
=s_!r^\ast q_!p^\ast\mathcal F.
$$
由 Cartesian square
$$
\begin{array}{ccc}
Z\times_Y Z' & \xrightarrow{\operatorname{pr}_2} & Z'\\
\downarrow{\scriptstyle \operatorname{pr}_1} & & \downarrow{\scriptstyle r}\\
Z & \xrightarrow{q} & Y
\end{array}
$$
的 base change，将 $r^\ast q_!$ 替换为 $\operatorname{pr}_{2!}\operatorname{pr}_1^\ast$，得到
$$
s_!\operatorname{pr}_{2!}\operatorname{pr}_1^\ast p^\ast\mathcal F
=(s\circ\operatorname{pr}_2)_!(p\circ\operatorname{pr}_1)^\ast\mathcal F.
$$
这正是复合 correspondence 定义的 functor。这里所有等号均表示由六函子 formalism 给出的自然同构。$\square$

## 本章小结

本附录固定了本书使用 quotient stack 处理 equivariant sheaves 的方式，并给出卷积 correspondence 的类型模板。后续所有 Hecke、Springer、Satake 和 Coulomb branch 构造都必须落入这个模板或显式说明偏离原因。

## 练习

**练习 A.1.** 证明若 $H$ 作用在自身上为左乘，则 $[H/H]\simeq\operatorname{Spec}k$。

**练习 A.2.** 令 $K\subset H$ 为 closed subgroup。写出 $H$-equivariant vector bundles on $H/K$ 与 $K$-representations 的等价，并比较它与命题 A.8。

**练习 A.3.** 对三个可复合 correspondence 写出两种加括号方式，并指出 associativity 需要哪一个 base change 同构。
