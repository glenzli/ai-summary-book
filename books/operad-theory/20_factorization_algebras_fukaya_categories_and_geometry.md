# 第二十章：Factorization algebra、Fukaya categories 与几何应用

本章说明 operad theory 如何进入几何。核心例子有两类：

1. factorization algebras and factorization homology；
2. Fukaya categories and higher operadic structures。

这些主题的完整证明依赖分析、层论、同伦论和辛几何。本章只建立严格的概念接口，并把深定理标为外部输入。

## 20.1 Disk categories

设 $\mathcal C^\otimes$ 是 presentable symmetric monoidal infinity-category，并且 tensor product separately preserves colimits。

**定义 20.1.** 令 $\mathbf{Disk}_n$ 为 $n$-维 disks 的 symmetric monoidal infinity-category：

1. 对象是有限个 $\mathbb R^n$ 的不交并
   $$
   \coprod_{i=1}^r\mathbb R^n,\qquad r\ge0;
   $$
2. morphisms 是 embeddings；
3. symmetric monoidal structure 由 disjoint union 给出。

若 $M$ 是 $n$-维拓扑或光滑流形，定义 overcategory
$$
\mathbf{Disk}_{n/M}=\mathbf{Disk}_n\times_{\mathbf{Mfld}_n}\mathbf{Mfld}_{n/M}.
$$
其对象可理解为带嵌入 $U\hookrightarrow M$ 的有限 disk 并。

**说明 20.2.** $\mathbf{Disk}_{n/M}$ 不是 ordinary poset of opens；morphisms 保留 embeddings 的同伦信息。若只取开集包含关系，会丢失同伦相干自同构。

## 20.2 Factorization algebras

**定义 20.3.** $M$ 上取值于 $\mathcal C$ 的 prefactorization algebra 是如下数据：

1. 对每个开集 $U\subset M$，给出对象 $\mathcal F(U)\in\mathcal C$；
2. 对任意两两不交开集 $U_1,\ldots,U_r\subset V$，给出结构映射
   $$
   \mathcal F(U_1)\otimes\cdots\otimes\mathcal F(U_r)\to\mathcal F(V);
   $$
3. 对 $r=0$，给出单位映射 $\mathbb 1_\mathcal C\to\mathcal F(V)$；
4. 对嵌套的不交开集配置，上述映射满足结合律、单位律和对称群等变性。

**定义 20.4.** Prefactorization algebra $\mathcal F$ 称为 factorization algebra，若它满足 Weiss descent：对每个开集 $V\subset M$ 和每个 Weiss cover $\{U_i\}_{i\in I}$，自然映射
$$
\operatorname{colim}_{(U_{i_1},\ldots,U_{i_r})\subset V}
\mathcal F(U_{i_1})\otimes\cdots\otimes\mathcal F(U_{i_r})
\longrightarrow
\mathcal F(V)
$$
是 equivalence。

这里 colimit 遍历 cover 中两两不交且并入 $V$ 的有限开集族。

**定义 20.5.** Factorization algebra $\mathcal F$ 称为 locally constant，若对任意 disks $D\subset D'\subset M$，包含诱导映射
$$
\mathcal F(D)\to\mathcal F(D')
$$
是 equivalence。

**外部输入定理 20.6.** 在 $\mathbb R^n$ 上，locally constant factorization algebras with values in $\mathcal C$ 与 $E_n$-algebras in $\mathcal C$ 之间存在 equivalence of infinity-categories：
$$
\operatorname{Fact}^{lc}_{\mathbb R^n}(\mathcal C)\simeq
\operatorname{Alg}_{E_n}(\mathcal C).
$$

**证明边界.** 一个 locally constant factorization algebra 的 disk 值给出对象 $A=\mathcal F(D)$；多个小 disk 嵌入大 disk 的结构映射给出 little disks operad 的运算。反向地，一个 $E_n$-algebra 可沿 disk embeddings 左 Kan extension 到 $\mathbb R^n$ 上的 factorization algebra。完整证明需要 Weiss descent、isotopy invariance 和 colimit 技术，本书作为外部输入。$\square$

## 20.3 Factorization homology

**定义 20.7.** 设 $A$ 是 $E_n$-algebra in $\mathcal C$，$M$ 是 $n$-manifold。$M$ 上以 $A$ 为系数的 factorization homology 定义为 colimit
$$
\int_M A
=
\operatorname{colim}_{(U\hookrightarrow M)\in\mathbf{Disk}_{n/M}} A(U),
$$
其中 $A$ 被视为 symmetric monoidal functor
$$
\mathbf{Disk}_n\to\mathcal C.
$$

若 $U\simeq\coprod_{i=1}^r\mathbb R^n$，则
$$
A(U)=A^{\otimes r}.
$$

**命题 20.8.** 对 $M=\mathbb R^n$，有 canonical equivalence
$$
\int_{\mathbb R^n}A\simeq A.
$$

**证明.** 在 $\mathbf{Disk}_{n/\mathbb R^n}$ 中，恒等嵌入 $\mathbb R^n\hookrightarrow\mathbb R^n$ 是 final object up to contractible choice：任意 disk 嵌入 $\mathbb R^n$ 可经同伦相干方式包含到整空间中。Colimit over a category with final object 等于该 final object 上的值。因此
$$
\int_{\mathbb R^n}A\simeq A(\mathbb R^n)=A.
$$
严格证明 finality 需要 embeddings 空间的 contractibility statement；此处使用 disks in Euclidean space 的标准同伦事实。$\square$

**外部输入定理 20.9（excision；AF-1）.** 若 $M$ 沿 collar 分解为
$$
M=M_-\cup_{N\times\mathbb R}M_+,
$$
其中 $N$ 是 $(n-1)$-manifold，则
$$
\int_M A
\simeq
\int_{M_-}A
\otimes_{\int_{N\times\mathbb R}A}
\int_{M_+}A.
$$
本书引用 Ayala--Francis, arXiv:1206.5522v6, Lemma 3.18 作为该 topological manifolds 版本的外部来源；分层或 Fukaya 版本需另行定位。

**说明 20.10.** Excision 是 factorization homology 的核心计算定理。它是 ordinary homology 的 Mayer-Vietoris 性质在 $E_n$-algebra 系数下的非交换版本。

**例 20.11.** 若 $n=1$ 且 $A$ 是 associative algebra，则
$$
\int_{S^1}A
$$
与 Hochschild homology $HH_\*(A)$ 对应。这是 factorization homology 与第十一、十二章 Hochschild 理论的连接。

该例的完整链级识别依赖 cyclic bar construction 和 $E_1$-algebra 的 factorization homology 计算，作为外部输入；本书引用 Ayala--Francis, arXiv:1206.5522v6, Theorem 3.19。

**说明 20.11.1.** 外部输入定理 N.18 和说明 N.19 给出本例的严格使用边界：圆周计算应写成
$$
\int_{S^1}A\simeq A\otimes^{\mathbf L}_{A\otimes A^{op}}A
$$
或等价的 cyclic bar construction。若没有指定 derived relative tensor product、边界版本和链级模型，不能把该式当作普通张量积公式。

## 20.4 Dunn additivity 与迭代代数

**外部输入定理 20.12（Dunn additivity）.** 在适当 symmetric monoidal infinity-category 中，有 equivalence of infinity-operads
$$
E_m\otimes E_n\simeq E_{m+n}.
$$
因此
$$
\operatorname{Alg}_{E_{m+n}}(\mathcal C)
\simeq
\operatorname{Alg}_{E_m}\big(\operatorname{Alg}_{E_n}(\mathcal C)\big).
$$

**说明 20.13.** 该定理说明 $E_{m+n}$-algebra 可看作 $E_m$-algebra object in $E_n$-algebras。它是许多“higher center”和 iterated Hochschild constructions 的 operadic 根源。

**命题 20.14.** 若 $A$ 是 $E_{m+n}$-algebra，则 $A$ canonically determines an $E_m$-algebra object in $\operatorname{Alg}_{E_n}(\mathcal C)$。

**证明.** 由外部输入定理 20.12，$E_{m+n}$-algebra 的结构等价于 $E_m\otimes E_n$-algebra 的结构。Tensor product of infinity-operads 的 universal property 把后者识别为 $E_m$-algebra object in $E_n$-algebras。$\square$

## 20.5 Fukaya categories as $A_\infty$-categories

**定义 20.15.** $A_\infty$-category $\mathcal A$ 由以下数据组成：

1. 对象类 $\operatorname{Ob}(\mathcal A)$；
2. 对每对对象 $X,Y$，给出链复形 $\operatorname{Hom}_\mathcal A(X,Y)$；
3. 对每个 $r\ge1$，给出次数 $r-2$ 的 composition maps
   $$
   m_r:\operatorname{Hom}(X_{r-1},X_r)\otimes\cdots\otimes
   \operatorname{Hom}(X_0,X_1)
   \to
   \operatorname{Hom}(X_0,X_r);
   $$
4. maps $m_r$ 满足 $A_\infty$ relations。

**定义 20.16.** Symplectic manifold $(X,\omega)$ 的 Fukaya category $\mathcal F(X)$ 是以合适 Lagrangian submanifolds 为对象、Floer complexes 为 morphism complexes、holomorphic polygons counts 为 $A_\infty$ compositions 的 $A_\infty$-category。

**警告 20.17.** 定义 20.16 是结构性描述，不是完整构造。完整 Fukaya category 需要选择 brane structures、gradings、spin structures、Novikov coefficients、transversality theory、compactness、bounding cochains 或 wrapped conditions。不同几何情形有不同模型。

**外部输入定理 20.18.** 在满足相应分析条件的 symplectic geometry 设置中，holomorphic polygon counts 定义 $A_\infty$-category，并且边界退化与 $A_\infty$ relations 对应。

**说明 20.19.** $A_\infty$ relations 的来源是 $1$-维 compactified moduli spaces 的边界。边界 strata 对应把一个 polygon 分裂成两个 polygons；代数上正是
$$
\sum m(\ldots,m(\ldots),\ldots)=0.
$$
这就是 operad 的 associahedra 与 Floer theory 的几何连接。

## 20.6 Operadic structures on Fukaya categories

Fukaya categories 不只形成单个 $A_\infty$-category。在不同几何操作下，它们还带有更高 operadic 结构。

**例 20.20（pair-of-pants product）.** 在某些 wrapped 或 exact 设置中，pair-of-pants 型曲面给出 Floer theory 上的乘法、coproduct 或 module operations。这些 operations 的 gluing 对应曲面 moduli 的 operadic composition。

**例 20.21（Swiss-cheese 型结构）.** 同时含 closed strings 与 open strings 的理论常由 Swiss-cheese operad 或其变体组织。Closed sector 通常带 $E_2$ 或 BV 型结构，open sector 带 $A_\infty$ 型结构，二者之间有兼容 action。

**外部输入定理 20.22.** 在特定几何假设下，Fukaya categories、wrapped Fukaya categories 或其 Hochschild invariants 可组织为由 surfaces、disks、stratified spaces 或 higher operads 控制的代数对象。

**证明边界.** 该定理不是单一通用结论。不同版本依赖不同的 compactness、gluing、orientation 和 transversality 定理。本书只记录 operadic 组织方式；具体几何定理必须在相应模型中单独引用。$\square$

## 20.7 Factorization homology 与 Fukaya 理论的接口

Factorization homology 把 $E_n$-algebra 沿 $n$-manifold 积分。Fukaya 理论把 symplectic 或 Liouville geometry 赋给 $A_\infty$-categories。二者的交汇点包括：

1. factorization homology of categories；
2. topological Fukaya categories；
3. wrapped Fukaya categories as cosheaves on skeleta；
4. Hochschild invariants and centers of Fukaya categories；
5. extended topological field theories。

**外部输入定理 20.23.** 在若干已验证设置中，Fukaya 型范畴可由局部模型通过 cosheaf/factorization homology gluing 得到；其 Hochschild 或 center 型不变量可由 factorization homology 计算。

**说明 20.24.** 这类定理是当前研究活跃区域。除非指定具体几何类别、系数、局部模型和 gluing 定理，本书不把它作为全局定理使用。

**说明 20.25.** 本章的计算性补充见定义 N.3、外部输入定理 N.15、外部输入定理 N.18 和外部输入定理 N.30；常见错误命题和不可混用约定见错误命题 O.23--正确边界 O.28。特别地，factorization homology 不等于普通同调，Fukaya category 的构造不由 operad 公理单独推出。

## 20.8 本章小结

Locally constant factorization algebras 是 $E_n$-algebras 的几何化；factorization homology 是把 $E_n$-algebra 沿 $n$-manifold 积分的 colimit。Dunn additivity 解释了迭代 $E_n$-结构。Fukaya categories 是 $A_\infty$-categories 的几何来源，并在更高结构下与 operads、factorization algebras 和 topological field theories 相连。所有涉及辛几何分析和全局 gluing 的结论都必须作为外部输入处理。

## 练习

**练习 20.1.** 写出 prefactorization algebra 的二重嵌套复合公理。

**练习 20.2.** 说明 locally constant factorization algebra on $\mathbb R$ 如何给出 associative algebra。

**练习 20.3.** 对 associative algebra $A$，解释 $\int_{S^1}A$ 与 cyclic bar construction 的关系。

**练习 20.4.** 写出 $A_\infty$-category 中 $m_1,m_2,m_3$ 参与的一条低阶关系。

**练习 20.5.** 说明为什么 Fukaya category 的构造不能只用形式 operad 公理完成，还需要分析定理。
