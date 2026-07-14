# 第十一章：Fukaya-Seidel category 与 Picard-Lefschetz theory

把 Landau--Ginzburg 势函数只当作一个多项式，会遗漏它沿复平面变化时产生的单值化。围绕临界值移动一次，正则纤维中的 vanishing cycle 经历 Dehn twist；选择一组有序 vanishing paths，又把这种几何组织成一个上三角的 $A_\infty$ 范畴。Fukaya--Seidel category 正是从 Lefschetz fibration 提取这一有向信息的模型。本章以第四章的多边形运算和第六章的非紧 exact 几何为基础，依次构造 thimbles、directed morphisms 与 Picard--Lefschetz 变换，并在 $\mathbb P^1$ 镜像上计算其 Kronecker 型代数。

## 11.1 Lefschetz fibration

**定义 11.1.** 一个 exact symplectic Lefschetz fibration 是映射
$$
\pi:E\to\mathbb C
$$
使得临界点非退化、临界值互异，并且在临界点附近可取复坐标使
$$
\pi(z_1,\ldots,z_n)=z_1^2+\cdots+z_n^2+\text{constant}.
$$
同时 $E$ 带 exact symplectic 结构并与 fibration 兼容。

**定义 11.2.** 取基点 $b\in\mathbb C$ 避开临界值。连接 $b$ 到临界值 $c_i$ 的嵌入路径 $\gamma_i$ 称为 vanishing path。沿 $\gamma_i$ 平行移动并在临界点塌缩得到 fiber $F_b$ 中的 vanishing cycle $V_i\subset F_b$ 和 total space 中的 Lefschetz thimble $\Delta_i\subset E$。

**命题 11.3.** vanishing cycle $V_i$ 是 $F_b$ 中的 Lagrangian sphere。

**证明.** 把临界值平移到 $0$，并在临界点附近采用 Lefschetz 局部模型
$\pi(z)=z_1^2+\cdots+z_n^2$。对足够小的正实数 $t$，纤维
$\pi^{-1}(t)$ 中的实点集
$$
V_t=\{(x_1,\ldots,x_n)\in\mathbb R^n:x_1^2+\cdots+x_n^2=t\}
$$
微分同胚于 $S^{n-1}$。标准辛形式
$\sum_j dx_j\wedge dy_j$ 在 $V_t\subset\{y_1=\cdots=y_n=0\}$ 上限制为
零，而 $\dim_{\mathbb R}V_t=n-1=\tfrac12\dim_{\mathbb R}\pi^{-1}(t)$，故
$V_t$ 是该纤维中的 Lagrangian sphere。沿 vanishing path 的辛平行移动是
纤维间的辛同构，因而把 $V_t$ 送到基点纤维中的 Lagrangian sphere
$V_i$。证毕。

## 11.2 Directed Fukaya category

**定义 11.4.** 给定有序 vanishing paths $\gamma_1,\ldots,\gamma_m$，directed Fukaya category $\mathcal A_{\mathrm{dir}}$ 的对象是 vanishing cycles $V_1,\ldots,V_m$，morphisms 定义为
$$
\operatorname{hom}(V_i,V_j)=
\begin{cases}
CF^\ast(V_i,V_j),& i<j,\\
k\cdot e_i,& i=j,\\
0,& i>j.
\end{cases}
$$
高阶复合由 fiber 中 holomorphic polygons 计数定义，并与方向条件相容。

**定义 11.5.** Fukaya-Seidel category $\mathcal F\mathcal S(\pi)$ 是 directed Fukaya category 的适当 twisted/split closure，或等价地由 Lefschetz thimbles 在 total space 中生成的 wrapped/partially wrapped category 的模型。具体版本依赖文献口径。

**外部输入定理 11.6（Seidel Fukaya-Seidel 构造）.** exact Lefschetz fibration 的 vanishing cycles 和 thimbles 构成 directed $A_\infty$ category；其 quasi-equivalence type 在 Hurwitz moves 和合适选择下按预期变化。
来源：Seidel, *Fukaya Categories and Picard-Lefschetz Theory*。

## 11.3 Picard-Lefschetz 变换

**定义 11.7.** 对 Lagrangian sphere $V\subset F$，Dehn twist $\tau_V$ 是 $F$ 的 compactly supported symplectomorphism，其局部模型由 cotangent bundle $T^\ast S^n$ 中的标准 twist 给出。

**外部输入定理 11.8（Picard-Lefschetz monodromy）.** 绕一个 Lefschetz 临界值的 symplectic monodromy Hamiltonian isotopic 于沿相应 vanishing cycle 的 Dehn twist。

**解释 11.9.** Picard-Lefschetz theory 把 base 中路径的 braid 操作翻译为 fiber Fukaya category 中的 Dehn twists。这是 exceptional collections、mutations 和 HMS 中 monodromy 对应的基础。

**外部输入定理 11.9A（Dehn twist exact triangle）.** 设 $V$ 是满足 Fukaya
分析包和分次假设的 Lagrangian sphere，$L$ 是另一 brane。则在
$D^\pi\mathcal F$ 中存在函子性 exact triangle
$$
HF^\ast(V,L)\otimes V\longrightarrow L\longrightarrow \tau_V(L)
\longrightarrow[1].
\tag{11.1}
$$
第一箭头是 evaluation。该三角把几何 Dehn twist 提升为范畴自函子；其构造
和 exactness 采用 Seidel 的 Picard--Lefschetz/Fukaya 理论作为外部输入。

## 11.4 HMS 形态

**定义 11.10.** 对 Fano variety $X$，其镜像 Landau-Ginzburg model 常写作
$$
W:Y\to\mathbb C.
$$
Fano/LG HMS 的标准 B-to-A 形态为
$$
\mathcal F\mathcal S(Y,W)\simeq\operatorname{Perf}(X).
$$

**命题 11.11.** 若 $\operatorname{Perf}_{\mathrm{dg}}(X)$ 由 exceptional
collection $(E_1,\ldots,E_m)$ split-generate，且
$\mathcal F\mathcal S(W)$ 由 vanishing thimbles
$(\Delta_1,\ldots,\Delta_m)$ split-generate，则只要构造 strictly unital
quasi-equivalence between the two directed full subcategories，并在对象上
匹配 $\Delta_i\mapsto E_i$，即可得到 Morita HMS。Pairwise morphism
complexes 为
$$
\operatorname{hom}_{\mathcal F\mathcal S}(\Delta_i,\Delta_j)
\quad\text{与}\quad
\mathbf R\operatorname{Hom}_X(E_i,E_j).
$$

**证明.** 两边生成对象给出 full directed subcategories；所假设的
quasi-equivalence 与两边 split-generation 满足命题 8.9，故得到 Morita
equivalence。若把有限 collection 压缩为直和的 endomorphism algebra，
必须保持投影到各 summand 的 orthogonal idempotents；只逐对比较上式的
cohomology groups 或 $\mu^2$ 不足以构造该 quasi-equivalence。证毕。

## 11.5 基本例子：$\mathbb P^1$

**例 11.12.** $\mathbb P^1$ 的镜像 potential 可取
$$
W(z)=z+qz^{-1}:\mathbb C^\ast\to\mathbb C.
$$
它有两个 critical points。Fukaya-Seidel category 有两个基本 thimbles；B-side $\operatorname{Perf}(\mathbb P^1)$ 由 exceptional collection
$$
(\mathcal O,\mathcal O(1))
$$
生成。两边 directed morphism algebra 的维数匹配：
$$
\dim\operatorname{Hom}(\mathcal O,\mathcal O(1))=2,
$$
对应两个交点或两条 thimble morphisms。

**命题 11.12A（B-side Kronecker 代数）.** 令
$G=\mathcal O\oplus\mathcal O(1)\in\operatorname{Perf}(\mathbb P^1)$。则
$\mathbf R\operatorname{End}(G)$ 形式化，并 quasi-isomorphic 于 Kronecker
quiver 的 path algebra：它有两个对象幂等元 $e_0,e_1$、两条从 $e_0$ 到
$e_1$ 的箭头，除此之外只有单位复合。

**证明.** 对 $i,j\in\{0,1\}$，
$$
\operatorname{Ext}^p(\mathcal O(i),\mathcal O(j))
\cong H^p(\mathbb P^1,\mathcal O(j-i)).
$$
$H^0(\mathcal O)=k$，$H^0(\mathcal O(1))=k^2$，而
$H^p(\mathcal O(-1))=0$ 及这里出现的所有 $p>0$ 上同调均为零。因此
endomorphism cohomology 集中在次数 $0$，两条非单位箭头恰来自
$H^0(\mathcal O(1))$ 的一组基。最小 $A_\infty$ 模型因次数和有向性没有
$\mu^d$（$d\ge3$）的非零值，故形式化。证毕。

**警告 11.13.** 例 11.12 的完整 HMS 证明还需比较乘法和 $A_\infty$ 结构，而不只是 morphism 维数。

Vanishing paths 的次序把 Lefschetz fibration 变成 directed $A_\infty$ 数据，绕临界值的 monodromy 则由相应 Lagrangian sphere 的 Dehn twist 表示。对 $z+qz^{-1}$，两个 thimbles 与 $(\mathcal O,\mathcal O(1))$ 都产生 Kronecker 型有向代数；一般 Fano/LG 情形沿同一路线比较 thimbles 与 exceptional objects，但必须连同高阶复合和生成性一起比较。

## 练习

**练习 11.1.** 对 $W(z)=z+qz^{-1}$，求 critical points 和 critical values。

**练习 11.2.** 写出 directed category 中 $i>j$ 时 morphism 为零如何影响可能的高阶复合。

**练习 11.3.** 解释 Dehn twist 与 Picard-Lefschetz monodromy 的关系。

**练习 11.4.** 证明 $\mathcal O,\mathcal O(1)$ split-generate $\operatorname{Perf}(\mathbb P^1)$ 可归约为 Beilinson 型分解，并标为外部输入。
