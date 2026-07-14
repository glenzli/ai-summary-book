# 第十九章：Rabinowitz Fukaya categories、singularities 与 matrix factorizations

奇点把镜像两侧同时推离最熟悉的范畴：B-side 上，$\mathrm D^b\operatorname{Coh}$ 的完美部分看不见奇异性，必须转向 singularity category 或 matrix factorizations；A-side 上，Milnor fiber 的 contact 边界和 Reeb dynamics 又促使 Rabinowitz 型而非普通 compact Fukaya 范畴出现。二者都含有“微分平方为势函数”的曲率痕迹，却不能因此被直接认作同一对象。本章从 Milnor fiber 与一个显式 matrix factorization 算起，再陈述 Brieskorn--Pham 情形的外部 HMS 结果，并只推出由 Morita 不变性严格保证的后果。

## 19.1 奇点与 Milnor fiber

**定义 19.1.** 设 $f:\mathbb C^{n+1}\to\mathbb C$ 在原点有孤立临界点。
取
$$
0<|\epsilon|\ll\delta\ll1,
$$
使 $f^{-1}(\epsilon)$ 与球面 $\partial B_\delta(0)$ 横截。Milnor fiber 定义为
$$
F_f=f^{-1}(\epsilon)\cap B_\delta(0).
$$
限制标准 Liouville form 并对边角作适当圆滑后，它带有 exact symplectic
domain 的自然模型；其 deformation type 与足够小的允许参数无关。

**定义 19.2.** Brieskorn-Pham polynomial 是
$$
f(x_0,\ldots,x_n)=x_0^{a_0}+\cdots+x_n^{a_n}
$$
形式的多项式。

**命题 19.2A（Brieskorn--Pham Milnor number）.** 若 $a_i\ge2$，则
$f=\sum_{i=0}^n x_i^{a_i}$ 在原点有孤立临界点，且其 Jacobian algebra 为
$$
\operatorname{Jac}(f)=
\frac{\mathbb C[x_0,\ldots,x_n]}
{(x_0^{a_0-1},\ldots,x_n^{a_n-1})}.
$$
因此 Milnor number
$$
\mu(f):=\dim_{\mathbb C}\operatorname{Jac}(f)
=\prod_{i=0}^n(a_i-1).
\tag{19.1}
$$

**证明.** $\partial f/\partial x_i=a_i x_i^{a_i-1}$，而 $a_i$ 在
$\mathbb C$ 中可逆，所以 Jacobian ideal 正是所示 monomial ideal。商的一组
基为
$$
x_0^{b_0}\cdots x_n^{b_n},\qquad 0\le b_i\le a_i-2.
$$
基向量数为显示的乘积，且各偏导的公共零点只有原点，故临界点孤立。证毕。

## 19.2 Matrix factorizations

**定义 19.3.** 令 $R=k[x_0,\ldots,x_n]$，$f\in R$。一个 finite-rank
matrix factorization 是有限秩自由 $R$-modules $E^0,E^1$ 与映射
$$
E^0\xrightarrow{d_0}E^1\xrightarrow{d_1}E^0
$$
满足 $d_1d_0=f\operatorname{id}_{E^0}$、
$d_0d_1=f\operatorname{id}_{E^1}$。这些对象及其 $\mathbb Z/2$-graded
commutator differential 构成 dg category
$$
\operatorname{MF}(\mathbb A^{n+1},f)
$$
；有对称群作用时还可取明确线性化的 equivariant 版本。

**例 19.3A（节点的秩一分解）.** 对 $R=k[x,y]$、$f=xy$，取
$$
E^0=R,\qquad E^1=R,\qquad d_0=x,\qquad d_1=y.
$$
则 $d_1d_0=xy=d_0d_1$，故得到一个 matrix factorization。把它忘成普通
$\mathbb Z/2$-复形会错误地要求 $xy=0$；正确做法是在势函数为 $f$ 的 curved
类别中保留 $d^2=f$。

**外部输入定理 19.4（Orlov singularity 关系）.** 对有限 Krull 维 regular
noetherian affine ring $R$ 与非零因子 $f\in R$，finite-rank matrix factorizations 的
homotopy category 在取相应 idempotent completion 后，与 hypersurface
$R/(f)$ 的 singularity category
$$
\mathrm D^b_{\mathrm{coh}}(R/(f))/\operatorname{Perf}(R/(f))
$$
等价。Equivariant、graded 或 nonaffine 版本需要分别指定，不能由本陈述
自动推出。来源：Orlov 的 Landau--Ginzburg/singularity category 定理。

## 19.3 Rabinowitz Fukaya category

**定义 19.5.** Rabinowitz Floer theory 研究 contact-type hypersurface 上的 Reeb dynamics 和 action functional 带 Lagrange multiplier 的 Floer theory。Rabinowitz Fukaya category 是把这种 Floer theory 范畴化后得到的 A-side 对象；具体模型依赖文献构造。

**警告 19.6.** Rabinowitz Fukaya category 不是 ordinary wrapped Fukaya category 的同义词。其对象、morphisms、grading 与完成方式必须按所调用的具体构造声明；本章只在定理 19.7 的来源模型中使用该记号。

**外部输入定理 19.7（Lekili-Ueda Brieskorn-Pham 结果）.** 对非 Calabi-Yau 型 Brieskorn-Pham singularities 的 Milnor fibers，Rabinowitz Fukaya categories 与 equivariant matrix factorizations 之间存在 HMS 型结果，并可用 Hochschild homology 计算 Rabinowitz Floer homology。
来源：Lekili-Ueda, *Homological mirror symmetry for Rabinowitz Fukaya categories of Milnor fibers of Brieskorn-Pham singularities*。

## 19.4 Categorical consequence

**命题 19.8.** 假设 Rabinowitz HMS 等价
$$
\mathcal R\mathcal F(F_f)\simeq\operatorname{MF}^{G}(\mathbb A^{n+1},f)
$$
在 Morita 意义下成立。则
$$
HH_\ast(\mathcal R\mathcal F(F_f))\cong
HH_\ast(\operatorname{MF}^{G}(\mathbb A^{n+1},f)).
$$

**证明.** 由 Hochschild homology 的 Morita invariance 直接得到。证毕。

**解释 19.9.** 若某个 Rabinowitz Floer invariant 可被 open-closed 或 Hochschild 结构识别，则 B-side matrix factorization 的 Hochschild homology 给出可计算模型。

在奇点语境中，$d^2=f$ 的 matrix factorization 精确记录了完美复形商掉以后仍存留的信息；Rabinowitz 范畴则引入 contact/Reeb 端的额外动力学，因此不能以 $\mathcal W(F_f)$ 代称。Lekili--Ueda 的结果在明确 Brieskorn--Pham 与等变假设下连接两者，Hochschild 同调同构是其 Morita 形式后果。超出这些假设时，正确的 A-side 模型与群作用仍是问题的一部分。

## 练习

**练习 19.1.** 对 $f=x^a+y^b$，写出其 Milnor fiber 的定义。

**练习 19.2.** 说明 matrix factorization 中 $d^2=f$ 与普通复形 $d^2=0$ 的差异。

**练习 19.3.** 证明命题 19.8。

**练习 19.4.** 解释为什么 Rabinowitz Fukaya category 需要单独声明模型，而不能直接写作 $\mathcal W(F_f)$。
