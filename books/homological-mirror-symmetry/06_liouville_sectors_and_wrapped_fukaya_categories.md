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

**定义 6.3（wrapped brane）.** 本章固定一个 Liouville completion
$\widehat M$、Maslov cover、background class 和系数域 $k$。Admissible
exact Lagrangian $L\subset\widehat M$ 是 properly embedded Lagrangian，
满足 $\lambda|_L=df_L$，并在圆柱端上与 compact Legendrian
$\Lambda\subset\partial M$ 的正锥一致：
$$
L\cap([R,\infty)\times\partial M)=[R,\infty)\times\Lambda.
$$
要求 chosen primitive $f_L$ 在每个 cylindrical end 上为常数。Wrapped
brane 还带相对于固定 Maslov cover 的 grading、相对于 background class
的 Pin/spin data，以及有限秩 $k$-局部系统。对象从一个固定
$\mathcal U$-小集合中选取。

## 6.2 Hamiltonian chords 与 wrapped morphisms

**定义 6.4（cofinal pair data）.** 对一对 wrapped branes，选择单调
cofinal Hamiltonians $H_i:[0,1]\times\widehat M\to\mathbb R$。在柱状端
$H_i(t,r,y)=h_i(r)$，且 $h_i'(r)=\tau_i>0$ 对大 $r$ 成立，其中
$\tau_i\to+\infty$ 并避开该对 Legendrian ends 的 Reeb-chord spectrum。
同时选择 contact-type $J_{i,t}$，并扰动到 Hamiltonian chords 非退化。
从 $L_0$ 到 $L_1$ 的 chord 是路径 $x:[0,1]\to\widehat M$，满足
$$
\dot x(t)=X_{H_i,t}(x(t)),\qquad
x(0)\in L_0,\quad x(1)\in L_1.
$$
线性-at-infinity 只是本章固定的模型；二次 Hamiltonian 模型的比较属于
外部输入定理 6.8。

**定义 6.5（有限斜率复形）.** 对固定 $i$，先定义 graded $k$-module
$$
CF^\ast(L_0,L_1;H_i)
=\bigoplus_{x\in\mathcal X(H_i;L_0,L_1)}
\operatorname{Hom}_k((E_{L_0})_{x(0)},(E_{L_1})_{x(1)})
\otimes o_x.
\tag{6.1}
$$
在外部输入定理 6.8 的 regular model 中，其 differential 由 rigid
Hamiltonian-perturbed strips 的 local-system parallel transport 与
orientation-line maps 之和定义。Exact wrapped theory 默认使用 $k$；
若另加 Novikov/action completion，必须重新声明 completed coefficient
module 和收敛拓扑。

**解释 6.6.** compact exact 情况中 morphisms 由交点生成；wrapped 情况中，正斜率 Hamiltonian 会在无穷远处沿 Reeb flow 包裹 Lagrangian，因此 morphisms 由 Hamiltonian chords 生成。Reeb dynamics 是 wrapped theory 的核心。

**定义 6.7（telescope wrapped complex）.** 对单调 continuation data，
记 continuation cochain maps 为
$$
\kappa_i:CF^\ast(L_0,L_1;H_i)\longrightarrow
CF^\ast(L_0,L_1;H_{i+1}).
$$
定义
$$
CW^\ast(L_0,L_1)
:=\operatorname*{hocolim}_{i}
\bigl(CF^\ast(L_0,L_1;H_i),\kappa_i\bigr),
\tag{6.2}
$$
即复形 dg category 中的标准 mapping telescope，并令
$HW^\ast=H^\ast(CW^\ast)$。在定理 6.8 的假设下，
$$
HW^\ast(L_0,L_1)\cong
\operatorname*{colim}_i HF^\ast(L_0,L_1;H_i).
\tag{6.3}
$$
对一列已严格选定的 cochain maps，raw colimit 当然仍是复形；在域上并
满足通常 exactness 条件时，它与 telescope 有相同 cohomology。但其
choice-independence 和后续 $A_\infty$ 运算仍需要 continuation higher
homotopies。故本书把 homotopy colimit (6.2) 作为 chain-level 定义，而只
在 cohomology 上写普通 colimit (6.3)。

**外部输入定理 6.8（pairwise wrapped analytic package）.** 设
$\widehat M$ 是 Liouville completion（sector 情形另要求 GPS 的 sectorial
convexity），$L_0,L_1$ 是定义 6.3 的 wrapped branes，并固定定义 6.4 的
nondegenerate cofinal data。假设选择了相容的 monotone continuation data
和 regular perturbations。则：

1. strip 与 continuation equations 是 Fredholm regular；固定 generators
   后有 action--energy estimate；
2. contact-type/sectorial maximum principle 给出 no-escape，exactness 按
   命题 E.4 排除非恒定 sphere 与单边 disk bubbles；
3. expected dimensions $0,1$ 的 compactifications 分别是有限定向
   $0$-manifolds 与以 broken strips/continuation trajectories 为边界的紧
   $1$-manifolds；
4. relative Pin determinant orientations 和局部系统 parallel transport
   使 (6.1) 的 differential、$\kappa_i$ 及 continuation homotopies
   良定义，并给出 $d^2=0$ 与 homotopy-coherent continuation system；
5. telescope (6.2) 的 quasi-isomorphism type 与 cofinal Hamiltonian
   family、contact-type almost complex structures 和 regular perturbation
   choices 无关，并满足 (6.3)。线性与常用二次 Hamiltonian models 在
   同样的 admissibility/maximum-principle 边界内 quasi-isomorphic。

该定理不由 exactness 单独推出：无穷远 $C^0$ 控制、Reeb-chord
nondegeneracy、compactness、orientation 和 continuation gluing 都是外部
分析输入。来源：wrapped Floer theory；Liouville sectors 与 choice-coherent
模型采用 Ganatra--Pardon--Shende, *Covariantly functorial wrapped Floer
theory on Liouville sectors*, arXiv:1706.03152。

## 6.3 Wrapped Fukaya category

**定义 6.9.** 在一套固定 coherent wrapped polygon/continuation model 中，
Liouville manifold $M$ 的 wrapped Fukaya category $\mathcal W(M)$ 是
$A_\infty$ category，其：

1. 对象为定义 6.3 的 wrapped branes；
2. morphism complexes 为 telescope complexes (6.2)；
3. 高阶复合由与所有 telescope/continuation levels 相容的 wrapped
   polygons（或等价 cascade/popsicle model）计数；每个 rigid polygon
   贡献 E.5 型的 orientation-line/local-system map；
4. universal perturbation data 在所有 broken boundary strata 上与较低阶
   polygon 和 continuation data 的 gluing limits 相容。

**外部输入定理 6.10（wrapped categorical package）.** 在定理 6.8 的
假设下，若 coherent universal wrapped polygon data 还满足统一
action--energy bounds、no-escape、regularity、oriented compactification
和 gluing，则定义 6.9 的运算次数为 $2-d$，其 suspended components 满足
精确恒等式 (B.3)。所得 $\mathcal W(M)$ cohomologically unital；改变上述
cofinal/coherent auxiliary data 给出 strictly unital models 之间的
quasi-equivalence（必要时先用定理 4.15 strictify）。

这里的“不变量性”只比较同一 Liouville/brane 几何上的辅助数据。Liouville
deformation、sector inclusion、stop removal 或对象集合改变分别需要额外
continuation/functoriality/generation theorem，不能由本定理自动推出。

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

**例 6.16.** 对 Weinstein manifold，critical handles 的 cocores 是 wrapped
Fukaya category 的基本对象。其精确生成范围采用外部输入定理 6.17，而
不是从 Weinstein handle decomposition 形式推出。

**外部输入定理 6.17（cocore/linking-disk split-generation）.** 设 $X$ 是
带固定 Weinstein handle presentation 的 Weinstein sector；在 partially
wrapped 情形再设 stop $\mathfrak f$ 是 GPS 意义下的 mostly Legendrian
stop。令 $\mathcal G$ 包含 critical handles 的 cocores，并在有 stop 时
加入其 Legendrian strata 的 linking disks。则
$$
\operatorname{thick}\{Y_G:G\in\mathcal G\}
=H^0\operatorname{Perf}(\mathcal W(X,\mathfrak f));
$$
等价地，$\mathcal W(X,\mathfrak f)_{\mathcal G}$ 的 inclusion 是 Morita
equivalence。该结论不声称 raw full subcategory 与整个 wrapped category
quasi-equivalent。来源：Ganatra--Pardon--Shende 的 sectorial
descent/generation theorem, arXiv:1809.03427。

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

**命题 6.18.** 若 $\mathcal W(M)$ 由对象集合 $\mathcal G$
split-generate，B-side $\mathcal B$ 由 $\mathcal H$ split-generate，且存在
strictly unital quasi-equivalence
$$
\mathcal W(M)_{\mathcal G}
\xrightarrow{\simeq_{\mathrm{qe}}}\mathcal B_{\mathcal H},
$$
则 $\mathcal W(M)\simeq_{\mathrm{Morita}}\mathcal B$。若两组对象有限并已
添加 finite direct sums，可改为比较 $\operatorname{End}(\bigoplus G_i)$ 与
$\operatorname{End}(\bigoplus H_i)$ 的 $A_\infty$ quasi-isomorphism，但须
保持对象 idempotents。

**证明.** 这是命题 8.9 的 wrapped 情况。wrapped 性只改变 A-side 的 morphism complexes 和复合定义，不改变 Morita 生成元比较的形式逻辑。证毕。

## 本章小结

wrapped Fukaya category 是非紧 Liouville 几何的 A-side 核心对象。它的 morphisms 来自 Hamiltonian chords，并通过 continuation maps 或 cofinal Hamiltonians 组织。Liouville sectors 使 wrapped categories 具有局部到整体性质，为 sectorial descent、stop removal 和 microlocal sheaf 模型提供技术基础。

## 练习

**练习 6.1.** 证明 Liouville vector field 在 Liouville domain 边界向外时，边界上的 $\lambda|_{\partial M}$ 是 contact form。

**练习 6.2.** 在 $T^\ast S^1$ 中描述 cotangent fiber 的 Hamiltonian chords。

**练习 6.3.** 解释为什么 continuation maps 对 wrapped morphisms 的定义必不可少。

**练习 6.4.** 给出 compact Fukaya category 与 wrapped Fukaya category 在对象和 morphisms 上的两个差异。
