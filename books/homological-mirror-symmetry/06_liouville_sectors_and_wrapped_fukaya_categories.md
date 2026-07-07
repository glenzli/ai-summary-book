# 第六章：Liouville manifolds、sectors 与 wrapped Fukaya categories

## 本章目标

本章引入非紧 A-side 的标准语言：Liouville manifolds、Liouville sectors、admissible Lagrangians、Hamiltonian chords 和 wrapped Fukaya categories。重点是说明 wrapped category 与 compact Fukaya category 的差异，并为 sectorial descent 和 stop removal 做准备。

## 依赖前置知识

需要第三章的 exact symplectic geometry、第四章的 Fukaya category 和第五章的能量过滤思想。

## 6.1 Liouville 几何

**定义 6.1.** Liouville manifold 是 exact symplectic manifold $(M,\lambda)$，其 Liouville vector field $Z$ 由
$$
\iota_Z d\lambda=\lambda
$$
定义，并满足在无穷远处向外完备。若 $M$ 是带边界紧流形且 $Z$ 沿边界向外横截，则称为 Liouville domain；完成化记为 $\widehat M$。

**定义 6.2.** Liouville domain 的边界 $\partial M$ 带 contact form $\alpha=\lambda|_{\partial M}$。Reeb vector field $R_\alpha$ 由
$$
\alpha(R_\alpha)=1,\qquad \iota_{R_\alpha}d\alpha=0
$$
定义。

**定义 6.3.** admissible exact Lagrangian 是 exact Lagrangian $L\subset\widehat M$，在圆柱端上与 Legendrian $\Lambda\subset\partial M$ 的正锥一致：
$$
L\cap([R,\infty)\times\partial M)=[R,\infty)\times\Lambda.
$$

## 6.2 Hamiltonian chords 与 wrapped morphisms

**定义 6.4.** 给定 Hamiltonian $H:\widehat M\to\mathbb R$，从 $L_0$ 到 $L_1$ 的 time-one Hamiltonian chord 是路径 $x:[0,1]\to\widehat M$，满足
$$
\dot x(t)=X_H(x(t)),\qquad x(0)\in L_0,\quad x(1)\in L_1.
$$

**定义 6.5.** wrapped Floer cochain complex $CW^\ast(L_0,L_1;H)$ 是由合适 Hamiltonian chords 生成的分次 $\Lambda$-向量空间，微分由带 Hamiltonian perturbation 的 Floer strips 计数定义。

**解释 6.6.** compact exact 情况中 morphisms 由交点生成；wrapped 情况中，正斜率 Hamiltonian 会在无穷远处沿 Reeb flow 包裹 Lagrangian，因此 morphisms 由 Hamiltonian chords 生成。Reeb dynamics 是 wrapped theory 的核心。

**定义 6.7.** 若选择一列斜率趋向无穷的 Hamiltonians $H_i$，wrapped morphism 可形式写作 direct limit
$$
CW^\ast(L_0,L_1)=\operatorname*{colim}_i CF^\ast(\varphi_{H_i}^1(L_0),L_1),
$$
其中 transition maps 是 continuation maps。严格模型也可直接使用二次 Hamiltonian 和 wrapped perturbation data。

**外部输入定理 6.8（wrapped Floer 不变量性）.** 在标准 Liouville/admissibility 假设下，$CW^\ast(L_0,L_1)$ 的 quasi-isomorphism type 不依赖 Hamiltonian cofinal family、almost complex structures 和 perturbation choices。  
来源：wrapped Floer theory 与 Ganatra-Pardon-Shende 的 Liouville sector 框架。

## 6.3 Wrapped Fukaya category

**定义 6.9.** Liouville manifold $M$ 的 wrapped Fukaya category $\mathcal W(M)$ 是 $A_\infty$ category，其：

1. 对象为 admissible exact Lagrangian branes；
2. morphism complexes 为 $CW^\ast(L_0,L_1)$；
3. 高阶复合由带 Hamiltonian perturbation 的 holomorphic polygons 计数给出；
4. continuation data 组织为 coherent system，使 $A_\infty$ 方程成立。

**外部输入定理 6.10（wrapped $A_\infty$ 方程）.** 在 standard wrapped Floer hypotheses 下，定义 6.9 的高阶复合满足 $A_\infty$ 方程，且 $\mathcal W(M)$ 在 quasi-equivalence 意义下为 Liouville 几何不变量。

**警告 6.11.** $\mathcal W(M)$ 通常比 compact Fukaya category 大。非紧 Lagrangians 和 Reeb chords 引入的 morphisms 会看到 $M$ 的无穷远 contact boundary。

## 6.4 Liouville sectors

**定义 6.12.** Liouville sector 是带边界 Liouville manifold，边界附近带有使 wrapped Floer theory 可控的凸性结构。直观上，它允许把 Liouville manifold 切成有边界的局部 pieces，并在这些 pieces 上定义协变的 wrapped categories。

**外部输入定理 6.13（Liouville sector functoriality）.** 对合适的 Liouville sector inclusion
$$
X\hookrightarrow Y
$$
存在协变 $A_\infty$ functor
$$
\mathcal W(X)\to\mathcal W(Y),
$$
且与 sectorial gluing 操作相容。  
来源：Ganatra-Pardon-Shende, *Covariantly functorial wrapped Floer theory on Liouville sectors*。

**解释 6.14.** 这个协变性与普通开子集上的层限制方向相反，更接近 cosheaf 行为。这正是 sectorial descent 的基础。

## 6.5 基本例子

**例 6.15.** 对 cotangent bundle $T^\ast Q$，cotangent fiber $T_q^\ast Q$ 是 admissible exact Lagrangian。wrapped Floer cohomology
$$
HW^\ast(T_q^\ast Q,T_q^\ast Q)
$$
与 based loop space chains 有深刻关系；本书后续只使用其作为生成直觉，具体定理列为外部输入。

**例 6.16.** 对 Weinstein manifold，critical handles 的 cocores 是 wrapped Fukaya category 的基本对象。GPS generation theorem 表明，在合适假设下 cocores split-generate $\mathcal W(M)$。

**外部输入定理 6.17（cocore generation）.** Weinstein sector 的 wrapped Fukaya category 由 critical handles 的 cocores 生成；带 mostly Legendrian stop 时，还需加入 linking disks。  
来源：Ganatra-Pardon-Shende 的 sectorial descent/generation 结果。

## 6.6 HMS 中的 wrapped 版本

wrapped HMS 常见形式为
$$
\mathcal W(M)\simeq\operatorname{Perf}(X)
$$
或
$$
\mathcal W(M)\simeq\operatorname{MF}(X,W),
$$
其中 $M$ 非紧，B-side 常为非适当或 Landau-Ginzburg 对象。wrapped category 的非紧性与 B-side 的非紧/奇异行为相匹配。

**命题 6.18.** 若 $\mathcal W(M)$ 由对象集合 $\mathcal G$ split-generate，且 B-side $\mathcal B$ 由 $\mathcal H$ split-generate，则证明 wrapped HMS 可归约为比较 full subcategories on $\mathcal G$ and $\mathcal H$ 的 endomorphism $A_\infty$ algebras。

**证明.** 这是命题 8.9 的 wrapped 情况。wrapped 性只改变 A-side 的 morphism complexes 和复合定义，不改变 Morita 生成元比较的形式逻辑。证毕。

## 本章小结

wrapped Fukaya category 是非紧 Liouville 几何的 A-side 核心对象。它的 morphisms 来自 Hamiltonian chords，并通过 continuation maps 或 cofinal Hamiltonians 组织。Liouville sectors 使 wrapped categories 具有局部到整体性质，为 sectorial descent、stop removal 和 microlocal sheaf 模型提供技术基础。

## 练习

**练习 6.1.** 证明 Liouville vector field 在 Liouville domain 边界向外时，边界上的 $\lambda|_{\partial M}$ 是 contact form。

**练习 6.2.** 在 $T^\ast S^1$ 中描述 cotangent fiber 的 Hamiltonian chords。

**练习 6.3.** 解释为什么 continuation maps 对 wrapped morphisms 的定义必不可少。

**练习 6.4.** 给出 compact Fukaya category 与 wrapped Fukaya category 在对象和 morphisms 上的两个差异。
